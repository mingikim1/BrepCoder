"""
BrepCoder model: B-rep encoder + projector + LLM.

    B-rep graph --[B-rep encoder]--> [N, 256] --[projector]--> [N, d] --+
                                                                       +--> LLM --> CAD code
    "<brep_start><brep><brep_end>" --[token embedding]-----------------+

- B-rep encoder: UV-Net style encoder pre-trained in Phase 1 (Phase1/brep_encoder.py). Each of
  the N face nodes is represented by concat(node_emb[128], graph_emb[128]) = 256 dims.
- Projector: Linear(256, 640) -> GELU -> Linear(640, 1024) -> GELU -> Linear(1024, d).
- LLM: Qwen2.5-1.5B (d = 1536). The <brep> token is replaced by the N projected node embeddings.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

from constants import BREP_TOKEN_INDEX, IGNORE_INDEX

# Phase1 and Phase2 both have a `dataset.py`, so the encoder is imported by file path
# instead of adding Phase1/ to sys.path.
_BREP_ENCODER_PATH = Path(__file__).resolve().parent.parent / "Phase1" / "brep_encoder.py"
_spec = importlib.util.spec_from_file_location("brep_encoder", _BREP_ENCODER_PATH)
_brep_encoder_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_brep_encoder_module)
BrepEncoderWrapper = _brep_encoder_module.BrepEncoderWrapper


def _load_matching(module: nn.Module, state_dict: Dict[str, torch.Tensor], name: str):
    """Load the tensors of `state_dict` whose name and shape match `module`."""
    own_state = module.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
    module.load_state_dict(matched, strict=False)
    print(f"Loaded {len(matched)}/{len(own_state)} {name} tensors")


class BrepCoderModel(nn.Module):

    def __init__(
        self,
        llm_model_path: str,
        code_hidden_size: int = 1536,
        freeze_llm: bool = False,
        freeze_encoder: bool = True,
        freeze_projection: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = False,
        use_gradient_checkpointing: bool = False,
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 3,
        graph_num_mlp_layers: int = 2,
    ):
        super().__init__()
        self.code_hidden_size = code_hidden_size
        self.freeze_llm = freeze_llm
        self.freeze_encoder = freeze_encoder
        self.freeze_projection = freeze_projection
        self.torch_dtype = torch_dtype

        # LLM
        print(f"Loading LLM from {llm_model_path}...")
        self.llm_config = AutoConfig.from_pretrained(llm_model_path, trust_remote_code=True)
        self.llm_hidden_size = self.llm_config.hidden_size
        self.vocab_size = self.llm_config.vocab_size
        if code_hidden_size != self.llm_hidden_size:
            raise ValueError(
                f"Projection output size ({code_hidden_size}) must match LLM hidden size ({self.llm_hidden_size}). "
                f"Set code_hidden_size to {self.llm_hidden_size}."
            )

        model_kwargs = {"config": self.llm_config, "torch_dtype": torch_dtype, "trust_remote_code": True}
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path, **model_kwargs)

        if use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
        if freeze_llm:
            self.llm.requires_grad_(False)

        # B-rep encoder (weights are loaded from the Phase 1 checkpoint via `load_brep_encoder`)
        self.brep_encoder = BrepEncoderWrapper(
            curve_channels=6,
            surface_channels=7,
            curve_emb_dim=128,
            surface_emb_dim=128,
            hidden_dim=graph_hidden_dim,
            output_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            num_mlp_layers=graph_num_mlp_layers,
            concat_graph_emb=True,
        )
        brep_output_dim = self.brep_encoder.output_dim

        # Projector (trained from scratch)
        self.brep_projection = nn.Sequential(
            nn.Linear(brep_output_dim, 640),
            nn.GELU(),
            nn.Linear(640, 1024),
            nn.GELU(),
            nn.Linear(1024, code_hidden_size),
        )

        self.brep_encoder.to(torch_dtype)
        self.brep_projection.to(torch_dtype)
        if freeze_encoder:
            self.brep_encoder.requires_grad_(False)
        if freeze_projection:
            self.brep_projection.requires_grad_(False)

    def encode_brep(self, graph: dgl.DGLGraph) -> List[torch.Tensor]:
        """Encode a batched B-rep graph into a list of per-graph node embeddings [N_i, code_hidden_size]."""
        brep_feats, padding_mask = self.brep_encoder(dgl.unbatch(graph))  # [B, max_N, 256], [B, max_N]
        num_nodes = padding_mask.sum(dim=1).tolist()
        node_feats = torch.cat([brep_feats[i, :n] for i, n in enumerate(num_nodes)], dim=0)
        projected = self.brep_projection(node_feats).to(self.torch_dtype)
        return list(torch.split(projected, num_nodes, dim=0))

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph: dgl.DGLGraph,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Replace each BREP_TOKEN_INDEX in `input_ids` with the node embeddings of the matching B-rep
        and left-pad the batch again. Inserted B-rep positions are excluded from the loss.
        """
        brep_features = self.encode_brep(graph)
        embed_tokens = self.llm.get_input_embeddings()

        new_embeds, new_masks, new_labels = [], [], []
        for b in range(input_ids.shape[0]):
            cur_ids, cur_mask, cur_brep = input_ids[b], attention_mask[b], brep_features[b]
            brep_positions = torch.where((cur_ids == BREP_TOKEN_INDEX) & (cur_mask == 1))[0].tolist()

            embeds, masks, labs = [], [], []
            prev = 0
            for pos in brep_positions:
                embeds += [embed_tokens(cur_ids[prev:pos]), cur_brep]
                masks += [cur_mask[prev:pos], cur_mask.new_ones(cur_brep.shape[0])]
                if labels is not None:
                    labs += [labels[b, prev:pos], labels.new_full((cur_brep.shape[0],), IGNORE_INDEX)]
                prev = pos + 1
            embeds.append(embed_tokens(cur_ids[prev:]))
            masks.append(cur_mask[prev:])
            if labels is not None:
                labs.append(labels[b, prev:])

            new_embeds.append(torch.cat(embeds, dim=0))
            new_masks.append(torch.cat(masks, dim=0))
            if labels is not None:
                new_labels.append(torch.cat(labs, dim=0))

        batch_size = len(new_embeds)
        max_len = max(e.shape[0] for e in new_embeds)
        inputs_embeds = new_embeds[0].new_zeros((batch_size, max_len, self.llm_hidden_size))
        padded_mask = attention_mask.new_zeros((batch_size, max_len))
        padded_labels = labels.new_full((batch_size, max_len), IGNORE_INDEX) if labels is not None else None
        for i in range(batch_size):
            seq_len = new_embeds[i].shape[0]
            inputs_embeds[i, -seq_len:] = new_embeds[i]
            padded_mask[i, -seq_len:] = new_masks[i]
            if labels is not None:
                padded_labels[i, -seq_len:] = new_labels[i]

        return inputs_embeds, padded_mask, padded_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph: dgl.DGLGraph,
        labels: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        """Returns `loss` (next-token cross-entropy on the code tokens), `logits` and the aligned `labels`."""
        inputs_embeds, attention_mask, labels = self.build_inputs_embeds(input_ids, attention_mask, graph, labels)
        hidden_states = self.llm.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False,
        )[0]
        logits = self.llm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)

        return SimpleNamespace(loss=loss, logits=logits, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, graph: dgl.DGLGraph, **generate_kwargs):
        """Generate CAD code tokens. Returns only the newly generated token ids."""
        inputs_embeds, attention_mask, _ = self.build_inputs_embeds(input_ids, attention_mask, graph)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)

    def load_brep_encoder(self, checkpoint_path: str):
        """Load the B-rep encoder pre-trained in Phase 1 (Phase1/output/best_model/brep_encoder.pth)."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Phase 1 B-rep encoder checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        missing, unexpected = self.brep_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded B-rep encoder from {checkpoint_path} (epoch: {checkpoint.get('epoch', 'n/a')})")
        if missing:
            print(f"  WARNING: {len(missing)} missing keys, e.g. {missing[:5]}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")

    def save_pretrained(self, save_path: str):
        """
        Save the model as:
            save_path/brep_encoder/pytorch_model.bin
            save_path/projection/pytorch_model.bin
            save_path/llm/{config.json, model.safetensors}
            save_path/config.json
        """
        for name, module in [("brep_encoder", self.brep_encoder), ("projection", self.brep_projection)]:
            os.makedirs(os.path.join(save_path, name), exist_ok=True)
            torch.save(module.state_dict(), os.path.join(save_path, name, "pytorch_model.bin"))

        llm_path = os.path.join(save_path, "llm")
        os.makedirs(llm_path, exist_ok=True)
        self.llm.config.save_pretrained(llm_path)
        llm_state_dict = self.llm.state_dict()
        # Qwen2.5-1.5B ties lm_head to the input embeddings; store the shared tensor only once.
        is_tied = (
            "lm_head.weight" in llm_state_dict and "model.embed_tokens.weight" in llm_state_dict
            and llm_state_dict["lm_head.weight"].data_ptr() == llm_state_dict["model.embed_tokens.weight"].data_ptr()
        )
        llm_state_dict = {
            k: v.detach().cpu().contiguous()
            for k, v in llm_state_dict.items()
            if not (is_tied and k == "lm_head.weight")
        }
        save_file(llm_state_dict, os.path.join(llm_path, "model.safetensors"), metadata={"format": "pt"})

        config = {
            "code_hidden_size": self.code_hidden_size,
            "llm_hidden_size": self.llm_hidden_size,
            "vocab_size": self.vocab_size,
            "freeze_llm": self.freeze_llm,
            "freeze_encoder": self.freeze_encoder,
            "freeze_projection": self.freeze_projection,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Model saved to {save_path}")

    def save_tokenizer(self, tokenizer, save_path: str):
        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

    @classmethod
    def from_pretrained(cls, model_path: str, llm_model_path: str, **kwargs) -> "BrepCoderModel":
        """Build the model from the base LLM at `llm_model_path` and load the weights saved by `save_pretrained`."""
        model = cls(llm_model_path=llm_model_path, **kwargs)

        llm_weights = os.path.join(model_path, "llm", "model.safetensors")
        encoder_weights = os.path.join(model_path, "brep_encoder", "pytorch_model.bin")
        projection_weights = os.path.join(model_path, "projection", "pytorch_model.bin")
        for path in [llm_weights, encoder_weights, projection_weights]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")

        _load_matching(model.llm, load_file(llm_weights), "LLM")
        _load_matching(model.brep_encoder, torch.load(encoder_weights, map_location="cpu"), "B-rep encoder")
        _load_matching(model.brep_projection, torch.load(projection_weights, map_location="cpu"), "projection")
        return model
