import argparse
import os
import json
from pathlib import Path
import torch
import deepspeed
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from tqdm import tqdm
import math
from datetime import datetime

from coca import DeepCADCoCa
from brep_encoder import BrepEncoderWrapper
from dataset import CoCaDataset, collate_fn
from loss import coca_loss_fn


def train_epoch(model_engine, dataloader, epoch, rank=0, log_file=None, log_interval=10):
    model_engine.train()
    total_loss = 0.0
    total_cap_loss = 0.0
    total_con_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for batch_idx, batch in enumerate(pbar):
        device = model_engine.device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        graphs = batch["graphs"]
        
        logits, brep_global_embed, text_global_embed, logit_scale = model_engine(
            graphs=graphs,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        total_loss_batch, loss_cap, loss_con = coca_loss_fn(
            logits=logits,
            brep_emb=brep_global_embed,
            text_emb=text_global_embed,
            logit_scale=logit_scale,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        model_engine.backward(total_loss_batch)
        model_engine.step()
        
        # Logging
        loss_val = total_loss_batch.item()
        total_loss += loss_val
        total_cap_loss += loss_cap.item()
        total_con_loss += loss_con.item()
        
        # Get global step from DeepSpeed
        global_step = model_engine.global_steps
        
        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "cap": f"{loss_cap.item():.4f}",
            "con": f"{loss_con.item():.4f}",
            "lr": f"{model_engine.get_lr()[0]:.2e}",
        })
        
        # Log to file every N steps
        if log_file is not None and rank == 0 and (batch_idx + 1) % log_interval == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(
                f"[{timestamp}] Epoch {epoch:3d} | Step {global_step:6d} | Batch {batch_idx+1:4d} | "
                f"Loss: {loss_val:.4f} | Cap: {loss_cap.item():.4f} | Con: {loss_con.item():.4f} | "
                f"LR: {model_engine.get_lr()[0]:.2e}\n"
            )
            log_file.flush()

    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0
    return total_loss / len(dataloader), total_cap_loss / len(dataloader), total_con_loss / len(dataloader)


def validate_epoch(model_engine, dataloader, epoch, rank=0, log_file=None):
    model_engine.eval()
    total_loss = 0.0
    total_cap_loss = 0.0
    total_con_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", disable=(rank != 0))
        
        for batch in pbar:
            device = model_engine.device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graphs = batch["graphs"]
            
            logits, brep_global_embed, text_global_embed, logit_scale = model_engine(
                graphs=graphs,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            total_loss_batch, loss_cap, loss_con = coca_loss_fn(
                logits=logits,
                brep_emb=brep_global_embed,
                text_emb=text_global_embed,
                logit_scale=logit_scale,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Logging
            loss_val = total_loss_batch.item()
            total_loss += loss_val
            total_cap_loss += loss_cap.item()
            total_con_loss += loss_con.item()
            
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "cap": f"{loss_cap.item():.4f}",
                "con": f"{loss_con.item():.4f}",
            })

    if len(dataloader) == 0:
        avg_loss = 0.0
        avg_cap = 0.0
        avg_con = 0.0
    else:
        avg_loss = total_loss / len(dataloader)
        avg_cap = total_cap_loss / len(dataloader)
        avg_con = total_con_loss / len(dataloader)
    
    # Log validation results
    if log_file is not None and rank == 0:
        global_step = model_engine.global_steps
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(
            f"[{timestamp}] Epoch {epoch:3d} | Step {global_step:6d} | [VALIDATION] | "
            f"Loss: {avg_loss:.4f} | Cap: {avg_cap:.4f} | Con: {avg_con:.4f} | "
            f"LR: {model_engine.get_lr()[0]:.2e}\n"
        )
        log_file.flush()
    
    return avg_loss, avg_cap, avg_con


def main():
    parser = argparse.ArgumentParser(description="Train CoCa model")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    parser = deepspeed.add_config_arguments(parser)
    
    # Data paths
    parser.add_argument("--code_dir", type=str, default="../../Dataset/code",
                       help="Directory containing code JSON files")
    parser.add_argument("--brep_dir", type=str, default="../../Dataset/graph",
                       help="Directory containing B-Rep graph bin files")
    parser.add_argument("--tokenizer_path", type=str,
                       default="../../Checkpoint/Qwen2.5-Coder-1.5B",
                       help="Path to tokenizer")
    parser.add_argument("--split_json", type=str, default=None,
                       help="Path to train_val_test_split.json (optional)")
    parser.add_argument("--vocab_map_path", type=str, default="./cad_vocab_map.json",
                       help="Path to vocab map JSON file (from build_vocab.py)")
    
    # Model parameters
    parser.add_argument("--brep_dim", type=int, default=128,
                       help="B-Rep feature dimension")
    parser.add_argument("--embed_dim", type=int, default=512,
                       help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8,
                       help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--max_nodes_limit", type=int, default=2048,
                       help="Upper limit for maximum number of nodes per graph (safety limit)")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    
    # Output paths
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    deepspeed.init_distributed()
    rank = deepspeed.comm.get_rank()
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("Loading Vocab Map...")
        
        # Create log file
        log_file_path = os.path.join(args.output_dir, "training_log.log")
        log_file = open(log_file_path, "w")
        # Write header
        log_file.write("=" * 100 + "\n")
        log_file.write("CoCa Training Log\n")
        log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 100 + "\n")
        log_file.write(
            f"{'Timestamp':<20} | {'Epoch':<6} | {'Step':<8} | {'Type':<12} | "
            f"{'Loss':<8} | {'Cap':<8} | {'Con':<8} | {'LR':<10}\n"
        )
        log_file.write("-" * 100 + "\n")
        log_file.flush()
        print(f"Log file created: {log_file_path}")
    else:
        log_file = None
    
    # Load vocab map
    with open(args.vocab_map_path, "r") as f:
        vocab_data = json.load(f)
        new_vocab_size = vocab_data["vocab_size"]
        new_to_old = {int(k): v for k, v in vocab_data["new_to_old"].items()}
        old_to_new = {int(k): v for k, v in vocab_data["old_to_new"].items()}
    
    if rank == 0:
        print(f"★ Reduced Vocab Size: {new_vocab_size} (Original: 150k+)")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Model Build (use reduced vocab size)
    brep_encoder = BrepEncoderWrapper(
        curve_channels=6, surface_channels=7, 
        curve_emb_dim=128, surface_emb_dim=128,  # Updated to match user's setting
        output_dim=args.brep_dim, hidden_dim=args.brep_dim,
        num_layers=3, num_mlp_layers=2,
        max_nodes_limit=args.max_nodes_limit,
        concat_graph_emb=True,  # Enable graph_emb concat (UV-Net style)
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    
    # Get actual output dimension from brep_encoder (accounts for concat_graph_emb)
    actual_brep_dim = brep_encoder.output_dim
    
    if rank == 0:
        print(f"BrepEncoder output dimension: {actual_brep_dim} (hidden_dim={args.brep_dim} + graph_emb_dim={args.brep_dim})")
    
    model = DeepCADCoCa(
        brep_encoder=brep_encoder,
        vocab_size=new_vocab_size,  # Use reduced vocab size
        brep_dim=actual_brep_dim,  # Use actual output dimension from brep_encoder
        embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout, max_seq_len=args.max_seq_len,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # 4. Dataset (pass vocab_map_path)
    train_dataset = CoCaDataset(
        code_dir=args.code_dir,
        brep_dir=args.brep_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split="train",
        split_json=args.split_json,
        vocab_map_path=args.vocab_map_path,
        cache_graphs=False  # Disable caching to avoid memory issues with multiprocessing
    )
    
    val_dataset = None
    if args.split_json and os.path.exists(args.split_json):
        val_dataset = CoCaDataset(
            code_dir=args.code_dir,
            brep_dir=args.brep_dir,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            split="val",
            split_json=args.split_json,
            vocab_map_path=args.vocab_map_path,
            cache_graphs=False  # Disable caching to avoid memory issues
        )
    
    if rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"Validation dataset size: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        if rank == 0:
            print("ERROR: Train dataset is empty!")
            print("Please check:")
            print(f"  - Code directory: {args.code_dir}")
            print(f"  - B-Rep directory: {args.brep_dir}")
            print(f"  - Split JSON: {args.split_json}")
        return

    # Calculate total steps and warmup steps based on dataset size
    # Load and modify DeepSpeed config dynamically
    ds_config_path = getattr(args, 'deepspeed_config', None)
    ds_config = None
    if ds_config_path and os.path.exists(ds_config_path):
        with open(ds_config_path, 'r') as f:
            ds_config = json.load(f)
        
        # Get batch size and gradient accumulation from config
        micro_batch_size = ds_config.get("train_micro_batch_size_per_gpu", 64)
        gradient_accumulation_steps = ds_config.get("gradient_accumulation_steps", 1)
        world_size = deepspeed.comm.get_world_size()
        
        # Steps per epoch
        steps_per_epoch = len(train_dataset) // (micro_batch_size * world_size * gradient_accumulation_steps)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        
        # Total steps
        total_num_steps = steps_per_epoch * args.num_epochs
        
        # Warmup steps (10% of total steps)
        warmup_ratio = 0.1
        warmup_num_steps = int(total_num_steps * warmup_ratio)
        
        if rank == 0:
            print(f"Dataset size: {len(train_dataset)}")
            print(f"Micro batch size per GPU: {micro_batch_size}")
            print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
            print(f"World size: {world_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total steps: {total_num_steps}")
            print(f"Warmup steps: {warmup_num_steps} ({warmup_ratio*100:.1f}% of total)")
        
        # Update scheduler params in config
        if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
            ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_num_steps
            ds_config["scheduler"]["params"]["total_num_steps"] = total_num_steps
            ds_config["scheduler"]["params"]["warmup_min_ratio"] = warmup_ratio
        
        # Save modified config temporarily or pass via args
        # DeepSpeed will read from file, so we need to update the file or pass via args
        import tempfile
        temp_config_path = os.path.join(args.output_dir, "ds_config_temp.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(temp_config_path, 'w') as f:
            json.dump(ds_config, f, indent=4)
        args.deepspeed_config = temp_config_path
        
        if rank == 0:
            print(f"Updated DeepSpeed config saved to: {temp_config_path}")

    # 5. Create DataLoader manually (DeepSpeed needs explicit DataLoader for custom collate_fn)
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    world_size = deepspeed.comm.get_world_size()
    rank = deepspeed.comm.get_rank()
    
    # Get batch size from config or use default
    batch_size = ds_config.get("train_micro_batch_size_per_gpu", 128) if ds_config else 128
    
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=ds_config.get("train_micro_batch_size_per_gpu", 16),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=8,          
        pin_memory=True,        
        prefetch_factor=2,      
        persistent_workers=True 
    )
    
    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=ds_config.get("train_micro_batch_size_per_gpu", 16),
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    # DeepSpeed initialize - config is provided via args.deepspeed_config (from command line)
    # Don't pass config parameter here to avoid conflict with command line argument
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=None
    )

    if rank == 0:
        print(f"DeepSpeed Engine Initialized.")
        print(f"Micro Batch Size: {model_engine.train_micro_batch_size_per_gpu()}")
        print(f"Train DataLoader size: {len(train_dataloader)}")
        
    # Track best validation loss for best model saving
    best_val_loss = float('inf')
    best_model_dir = os.path.join(args.output_dir, "best_model")
    if rank == 0:
        os.makedirs(best_model_dir, exist_ok=True)
        
    # 7. Training Loop
    for epoch in range(args.num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        if hasattr(train_dataloader, 'data_sampler') and hasattr(train_dataloader.data_sampler, 'set_epoch'):
            train_dataloader.data_sampler.set_epoch(epoch)
        elif hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
            
        avg_loss, avg_cap, avg_con = train_epoch(
            model_engine, train_dataloader, epoch+1, rank, 
            log_file=log_file, log_interval=10
        )
        
        if rank == 0:
            print(f"Train Loss: {avg_loss:.4f} (Cap: {avg_cap:.4f}, Con: {avg_con:.4f})")
            # Log epoch average
            global_step = model_engine.global_steps
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(
                f"[{timestamp}] Epoch {epoch+1:3d} | Step {global_step:6d} | [TRAIN AVG] | "
                f"Loss: {avg_loss:.4f} | Cap: {avg_cap:.4f} | Con: {avg_con:.4f} | "
                f"LR: {model_engine.get_lr()[0]:.2e}\n"
            )
            log_file.flush()
        
        # Validation
        val_loss = None
        if val_dataloader is not None:
            val_loss, val_cap, val_con = validate_epoch(
                model_engine, val_dataloader, epoch+1, rank, log_file=log_file
            )
            if rank == 0:
                print(f"Val Loss: {val_loss:.4f} (Cap: {val_cap:.4f}, Con: {val_con:.4f})")
        
        # Save weights (PyTorch only, no DeepSpeed checkpoint for speed)
        if rank == 0:
            # Extract the actual model from DeepSpeed engine
            model_state = model_engine.module.state_dict()
            
            # Separate brep_encoder weights
            brep_encoder_state = {}
            coca_state = {}
            gen_attn_pool_state = {}  # Generative attention pooling weights (before multimodal decoder)
            
            for key, value in model_state.items():
                if key.startswith('brep_encoder.'):
                    # Remove 'brep_encoder.' prefix for cleaner state dict
                    brep_key = key[len('brep_encoder.'):]
                    brep_encoder_state[brep_key] = value
                else:
                    coca_state[key] = value
                    # Also extract gen_pool weights (generative attention pooling)
                    if key.startswith('gen_pool_query') or key.startswith('gen_pool_attn.') or key.startswith('gen_pool_norm.'):
                        gen_attn_pool_state[key] = value
            
            # Save epoch checkpoint (if save_every matches)
            if (epoch + 1) % args.save_every == 0:
                epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
                os.makedirs(epoch_dir, exist_ok=True)
                
                # Save brep_encoder weights separately
                brep_ckpt_path = os.path.join(epoch_dir, "brep_encoder.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': brep_encoder_state,
                }, brep_ckpt_path)
                
                # Save CoCa model weights (without brep_encoder)
                coca_ckpt_path = os.path.join(epoch_dir, "coca_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': coca_state,
                }, coca_ckpt_path)
                
                # Also save full model for convenience
                full_ckpt_path = os.path.join(epoch_dir, "full_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model_state,
                }, full_ckpt_path)
                
                # Save generative attention pooling weights (before multimodal decoder)
                gen_attn_pool_path = os.path.join(epoch_dir, "gen_attn_pool.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': gen_attn_pool_state,
                }, gen_attn_pool_path)
                
                print(f"Saved checkpoints to {epoch_dir}:")
                print(f"  - B-Rep Encoder: {brep_ckpt_path}")
                print(f"  - CoCa Model: {coca_ckpt_path}")
                print(f"  - Full Model: {full_ckpt_path}")
                print(f"  - Gen Attn Pool: {gen_attn_pool_path}")
            
            # Save best model based on validation loss
            if val_loss is not None and val_loss < best_val_loss:
                old_best = best_val_loss if best_val_loss != float('inf') else None
                best_val_loss = val_loss
                if old_best is not None:
                    print(f"New best model! Val Loss: {val_loss:.4f} (previous best: {old_best:.4f})")
                else:
                    print(f"New best model! Val Loss: {val_loss:.4f}")
                
                # Save best model
                brep_best_path = os.path.join(best_model_dir, "brep_encoder.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'state_dict': brep_encoder_state,
                }, brep_best_path)
                
                coca_best_path = os.path.join(best_model_dir, "coca_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'state_dict': coca_state,
                }, coca_best_path)
                
                full_best_path = os.path.join(best_model_dir, "full_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'state_dict': model_state,
                }, full_best_path)
                
                # Save generative attention pooling weights (before multimodal decoder)
                gen_attn_pool_best_path = os.path.join(best_model_dir, "gen_attn_pool.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'state_dict': gen_attn_pool_state,
                }, gen_attn_pool_best_path)
                
                print(f"Saved best model to {best_model_dir}:")
                print(f"  - B-Rep Encoder: {brep_best_path}")
                print(f"  - CoCa Model: {coca_best_path}")
                print(f"  - Full Model: {full_best_path}")
                print(f"  - Gen Attn Pool: {gen_attn_pool_best_path}")
    
    # Close log file
    if rank == 0 and log_file is not None:
        log_file.write("=" * 100 + "\n")
        log_file.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 100 + "\n")
        log_file.close()
        print(f"\nTraining completed. Log file saved to: {os.path.join(args.output_dir, 'training_log.log')}")

if __name__ == "__main__":
    main()