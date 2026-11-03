# loss.py
import torch
import torch.nn.functional as F
import torch.distributed as dist

def coca_loss_fn(logits, brep_emb, text_emb, logit_scale, input_ids, attention_mask):

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Use clone to avoid inplace modification of input_ids
    shift_labels = shift_labels.clone().masked_fill(shift_mask == 0, -100)

    loss_cap = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    # Normalize embeddings
    brep_emb = F.normalize(brep_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    
    local_batch_size = brep_emb.size(0)
    
    # Gather from all GPUs if using distributed training
    # Use torch.distributed.nn.functional.all_gather for gradient preservation (like 11_Project)
    if dist.is_initialized():
        from torch.distributed.nn.functional import all_gather
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Gather embeddings from all GPUs (preserves gradients)
        # This is faster than dist.all_gather + manual gradient handling
        brep_emb_list = all_gather(brep_emb)  # Returns list of tensors with gradients
        text_emb_list = all_gather(text_emb)  # Returns list of tensors with gradients
        
        global_brep_emb = torch.cat(brep_emb_list, dim=0)  # [global_batch_size, d_model]
        global_text_emb = torch.cat(text_emb_list, dim=0)  # [global_batch_size, d_model]
        
        global_batch_size = global_brep_emb.size(0)
        
        # Compute similarity matrix with global batch
        # Use logit_scale as temperature (divide instead of multiply for numerical stability)
        # logits = similarity / temperature, where temperature = 1.0 / logit_scale
        temperature = 1.0 / logit_scale.clamp(min=1e-8)  # Prevent division by zero
        
        # logits_i2t: [local_batch_size, global_batch_size]
        # Each row corresponds to a local brep_emb, columns are all global text_emb
        logits_i2t = torch.mm(brep_emb, global_text_emb.t()) / temperature
        
        # logits_t2i: [local_batch_size, global_batch_size]
        # Each row corresponds to a local text_emb, columns are all global brep_emb
        logits_t2i = torch.mm(text_emb, global_brep_emb.t()) / temperature
        
        # Labels: diagonal elements are positive pairs
        # For each local batch, positive pairs are offset by rank * local_batch_size
        labels = torch.arange(rank * local_batch_size, rank * local_batch_size + local_batch_size, 
                             device=logits_i2t.device)
        
        # logits_i2t and logits_t2i are already for local batch, no need to slice
        # They have shape [local_batch_size, global_batch_size]
        # Compute cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
    else:
        # Local batch mode (no distributed training)
        batch_size = local_batch_size
        
        # Compute similarity matrix
        # Use logit_scale as temperature
        temperature = 1.0 / logit_scale.clamp(min=1e-8)
        logits_i2t = torch.mm(brep_emb, text_emb.t()) / temperature
        logits_t2i = torch.mm(text_emb, brep_emb.t()) / temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits_i2t.device)
        
        # Compute cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
    
    loss_con = (loss_i2t + loss_t2i) / 2

    total_loss = 2.0 * loss_cap + 1.0 * loss_con

    return total_loss, loss_cap, loss_con