import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
    
class DeepCADCoCa(nn.Module):
    def __init__(self,
                 brep_encoder,
                 vocab_size,
                 brep_dim=256,  # Updated: 128 (node_emb) + 128 (graph_emb) = 256
                 embed_dim=512,
                 num_heads=8,
                 num_layers=8,
                 dropout=0.1,
                 max_seq_len=2048,
                 use_gradient_checkpointing=True):  # Enable gradient checkpointing for memory efficiency
        super().__init__()

        self.brep_encoder = brep_encoder
        self.embed_dim = embed_dim
        self.split_layer = num_layers // 2
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.brep_proj = nn.Linear(brep_dim, embed_dim)

        self.gen_pool_query = nn.Parameter(torch.randn(1, 256, embed_dim))
        self.gen_pool_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gen_pool_norm = LayerNorm(embed_dim)

        self.con_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.con_pool_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.con_pool_norm = LayerNorm(embed_dim)

        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Unimodal Decoder
        self.unimodal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim,
                                       nhead=num_heads,
                                       dim_feedforward=embed_dim * 4,
                                       dropout=dropout,
                                       batch_first=True,
                                       norm_first=True)
            for _ in range(self.split_layer)
        ])
        self.unimodal_norm = LayerNorm(embed_dim)
    
        # Multimodal Decoder
        self.multimodal_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim,
                                       nhead=num_heads,
                                       dim_feedforward=embed_dim * 4,
                                       dropout=dropout,
                                       batch_first=True,
                                       norm_first=True)
            for _ in range(num_layers - self.split_layer)
        ])
        self.multimodal_norm = LayerNorm(embed_dim)

        # Head
        self.output_head = nn.Linear(embed_dim, vocab_size)
        # Logit scale parameterized as exp(learnable) to ensure it stays positive
        # Initialize to correspond to temperature ~0.07 (like 11_Project)
        # logit_scale = exp(learnable_param), so temperature = 1.0 / logit_scale
        # For temperature=0.07, logit_scale = 1.0/0.07 ≈ 14.3, so learnable_param = log(14.3) ≈ 2.66
        # Clamp logit_scale to reasonable range [1.0, 100.0] -> temperature [0.01, 1.0]
        self.logit_scale_param = nn.Parameter(torch.ones([]) * 2.66)  # log(14.3)

        self.__init_weights()

    def __init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.gen_pool_query, std=0.02)
        nn.init.trunc_normal_(self.con_pool_query, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, graphs, input_ids, attention_mask):
        brep_feats, brep_padding_mask = self.brep_encoder(graphs)

        batch_size = brep_feats.size(0)

        # B-Rep Pipeline
        brep_emb = self.brep_proj(brep_feats)
        brep_key_padding_mask = (brep_padding_mask == 0).bool()

        gen_query = self.gen_pool_query.expand(batch_size, -1, -1)
        gen_feats, _ = self.gen_pool_attn(gen_query, brep_emb, brep_emb, key_padding_mask=brep_key_padding_mask)
        gen_feats = self.gen_pool_norm(gen_feats)

        con_query = self.con_pool_query.expand(batch_size, -1, -1)
        brep_global_embed, _ = self.con_pool_attn(con_query, gen_feats, gen_feats)
        brep_global_embed = self.con_pool_norm(brep_global_embed).squeeze(1)

        # Text Pipeline
        seq_len = input_ids.size(1)
        text_emb = self.text_embed(input_ids) + self.pos_embed[:, :seq_len, :]

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'), diagonal=1)
        causal_mask = causal_mask.to(dtype=text_emb.dtype)

        # Use bool dtype to match causal_mask expectations (fixes deprecation warning)
        text_key_padding_mask = (attention_mask == 0).bool()

        # Unimodal Decoder
        x = text_emb
        for layer in self.unimodal_layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, causal_mask, text_key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, src_mask=causal_mask, src_key_padding_mask=text_key_padding_mask)
        x_unimodal = self.unimodal_norm(x)

        # Contrastive Embed (EOS)
        eos_indices = attention_mask.sum(1).long() - 1
        eos_indices = eos_indices.clamp(min=0)

        text_global_embed = x_unimodal[torch.arange(batch_size), eos_indices]

        # Multimodal Decoder
        x_multi = x_unimodal
        for layer in self.multimodal_layers:
            if self.use_gradient_checkpointing and self.training:
                # Create a wrapper function to avoid lambda late binding issue
                def create_decoder_forward(decoder_layer):
                    def forward_fn(tgt, mem, mask, key_mask):
                        return decoder_layer(tgt, memory=mem, tgt_mask=mask, tgt_key_padding_mask=key_mask)
                    return forward_fn
                
                x_multi = checkpoint(
                    create_decoder_forward(layer),
                    x_multi, gen_feats, causal_mask, text_key_padding_mask,
                    use_reentrant=False
                )
            else:
                x_multi = layer(x_multi, memory=gen_feats, tgt_mask=causal_mask, tgt_key_padding_mask=text_key_padding_mask)
        x_multimodal = self.multimodal_norm(x_multi)

        logits = self.output_head(x_multimodal)

        # Clamp logit_scale to reasonable range [1.0, 100.0] to prevent numerical issues
        # This corresponds to temperature range [0.01, 1.0]
        # For temperature=0.07, logit_scale = 1.0/0.07 ≈ 14.3 (within range)
        logit_scale = torch.clamp(torch.exp(self.logit_scale_param), min=1.0, max=100.0)
        
        return logits, brep_global_embed, text_global_embed, logit_scale