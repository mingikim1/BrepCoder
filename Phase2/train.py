"""
Phase 2 training: B-rep graph -> Python-like CAD code (reverse engineering).

Usually launched through `scripts/train.sh`. Direct usage:
    deepspeed --num_gpus=4 train.py --deepspeed ds_config/zero2.json --code_dir ... --graph_dir ... \
        --split_json ... --llm_model_path ... --output_dir ...
    torchrun --nproc_per_node=4 train.py ...    # DDP without DeepSpeed
    python train.py ...                         # single device

Checkpoints are written to `output_dir/{checkpoint_epoch_N, best_model, final_model}`.
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from constants import IGNORE_INDEX
from dataset import BrepCodeDataset, collate_fn, prepare_tokenizer
from model import BrepCoderModel


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def create_optimizer_grouped_params(model, args):
    """Split the trainable parameters into encoder / projector / LLM groups, each with its own learning rate."""
    groups = {"encoder": [], "projector": [], "llm": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "brep_encoder" in name:
            groups["encoder"].append(param)
        elif "brep_projection" in name:
            groups["projector"].append(param)
        else:
            groups["llm"].append(param)

    lrs = {"encoder": args.encoder_lr, "projector": args.projector_lr, "llm": args.llm_lr}
    param_groups = []
    print("Optimizer parameter groups:")
    for name, params in groups.items():
        if params:
            param_groups.append({"params": params, "lr": lrs[name], "weight_decay": args.weight_decay, "name": name})
            print(f"  {name}: {sum(p.numel() for p in params):,} params, lr={lrs[name]}")
    return param_groups


def load_deepspeed_config(config_path, args, world_size, num_training_steps, num_warmup_steps):
    """Fill the DeepSpeed config with the values given on the command line."""
    with open(config_path, "r") as f:
        ds_config = json.load(f)

    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * world_size
    ds_config["gradient_clipping"] = args.max_grad_norm
    ds_config["bf16"]["enabled"] = args.bf16
    if args.bf16:
        ds_config["fp16"]["enabled"] = False

    if "optimizer" in ds_config:
        # Default lr only; every parameter group sets its own lr.
        ds_config["optimizer"]["params"]["lr"] = args.encoder_lr
        ds_config["optimizer"]["params"]["weight_decay"] = args.weight_decay
    if "scheduler" in ds_config:
        ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
        ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    return ds_config


def token_accuracy(outputs):
    """Number of correctly predicted next tokens and number of target tokens."""
    predictions = outputs.logits[:, :-1].argmax(dim=-1)
    labels = outputs.labels[:, 1:]
    valid = labels != IGNORE_INDEX
    return (predictions.eq(labels) & valid).sum().item(), valid.sum().item()


def log_mlflow(metrics, step):
    try:
        for key, value in metrics.items():
            if value is not None and math.isfinite(value):
                mlflow.log_metric(key, value, step=step)
    except Exception:
        pass


def train_epoch(model, dataloader, epoch, args, device, optimizer, scheduler=None,
                use_deepspeed=False, rank=0, use_mlflow=False, global_step=0):
    """
    Train for one epoch and return the mean loss / token accuracy.
    With DeepSpeed, `model` is the DeepSpeed engine, which owns the optimizer and scheduler.
    """
    model.train()
    total_loss, num_micro_batches = 0.0, 0
    total_correct, total_tokens = 0, 0
    step = global_step

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
    for batch_idx, batch in enumerate(pbar):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            graph=batch["graph"],
            labels=batch["labels"].to(device),
        )
        loss = outputs.loss

        if use_deepspeed:
            model.backward(loss)
            model.step()  # DeepSpeed applies the update only at gradient accumulation boundaries.
        else:
            (loss / args.gradient_accumulation_steps).backward()

        total_loss += loss.item()
        num_micro_batches += 1
        correct, tokens = token_accuracy(outputs)
        total_correct += correct
        total_tokens += tokens

        is_boundary = (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader)
        if not is_boundary:
            continue

        if use_deepspeed:
            grad_norm = model.get_global_grad_norm() or 0.0
        else:
            max_norm = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        step += 1

        lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)}
        acc = correct / tokens if tokens > 0 else 0.0
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}", "grad": f"{grad_norm:.2f}",
                              "lr": f"{next(iter(lrs.values())):.2e}"})
            if use_mlflow:
                metrics = {"train_loss": loss.item(), "train_accuracy": acc, "grad_norm": grad_norm}
                metrics.update({f"learning_rate_{name}": lr for name, lr in lrs.items()})
                log_mlflow(metrics, step)

    return {
        "loss": total_loss / max(num_micro_batches, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "steps": step - global_step,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Validation loss / token accuracy, averaged over all ranks."""
    model.eval()
    stats = torch.zeros(4, dtype=torch.float64, device=device)  # loss sum, batches, correct, tokens
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            graph=batch["graph"],
            labels=batch["labels"].to(device),
        )
        correct, tokens = token_accuracy(outputs)
        stats += torch.tensor([outputs.loss.item(), 1, correct, tokens], dtype=torch.float64, device=device)

    if dist.is_initialized():
        dist.all_reduce(stats)
    loss_sum, num_batches, correct, tokens = stats.tolist()
    return {"loss": loss_sum / max(num_batches, 1), "token_accuracy": correct / max(tokens, 1)}


def init_mlflow(args, world_size):
    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        username = os.environ.get("MLFLOW_USERNAME", "")
        password = os.environ.get("MLFLOW_PASSWORD", "")
        if username and password:
            from urllib.parse import quote_plus
            uri = tracking_uri.rstrip("/")
            if not uri.startswith(("http://", "https://")):
                uri = "http://" + uri
            tracking_uri = uri.replace("://", f"://{quote_plus(username)}:{quote_plus(password)}@", 1)
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.start_run(run_name=args.mlflow_run_name)
    params = {k: v for k, v in vars(args).items() if not k.startswith("mlflow")}
    params["num_gpus"] = world_size
    params["effective_batch_size"] = args.batch_size * args.gradient_accumulation_steps * world_size
    mlflow.log_params(params)


def setup_logger(output_dir, is_main):
    logger = logging.getLogger("BrepCoder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if is_main:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        for handler in [logging.FileHandler(log_file), logging.StreamHandler()]:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.info(f"Training log: {log_file}")
    return logger


def train(args):
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    use_deepspeed = args.deepspeed is not None
    if use_deepspeed and not DEEPSPEED_AVAILABLE:
        raise ImportError("--deepspeed was given but DeepSpeed is not installed.")
    use_mlflow = args.use_mlflow and MLFLOW_AVAILABLE and is_main

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, is_main)
    if is_main:
        logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
        logger.info(f"World size: {world_size} | DeepSpeed: {use_deepspeed} | Device: {device}")

    if use_mlflow:
        try:
            init_mlflow(args, world_size)
        except Exception as e:
            print(f"Warning: failed to initialize MLflow: {e}")
            use_mlflow = False

    # Data
    tokenizer = prepare_tokenizer(args.llm_model_path)
    dataset_kwargs = dict(code_dir=args.code_dir, graph_dir=args.graph_dir, split_json=args.split_json,
                          tokenizer=tokenizer, max_length=args.max_length)
    train_dataset = BrepCodeDataset(split="train", **dataset_kwargs)
    val_dataset = BrepCodeDataset(split="val", **dataset_kwargs) if args.do_eval else None

    num_workers = 0 if world_size > 1 else args.num_workers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
        )

    # Model
    model = BrepCoderModel(
        llm_model_path=args.llm_model_path,
        code_hidden_size=args.code_hidden_size,
        freeze_llm=args.freeze_llm,
        freeze_encoder=args.freeze_encoder,
        freeze_projection=args.freeze_projection,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_flash_attn=args.use_flash_attn,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_num_layers=args.graph_num_layers,
        graph_num_mlp_layers=args.graph_num_mlp_layers,
    )
    if args.brep_encoder_checkpoint and os.path.exists(args.brep_encoder_checkpoint):
        model.load_brep_encoder(args.brep_encoder_checkpoint)
    else:
        print(f"WARNING: Phase 1 B-rep encoder checkpoint not found ({args.brep_encoder_checkpoint}). "
              f"The B-rep encoder is randomly initialized.")
        if args.freeze_encoder:
            print("WARNING: The randomly initialized B-rep encoder is frozen (--freeze_encoder) and will not be trained.")

    if len(tokenizer) > model.llm.config.vocab_size:
        model.llm.resize_token_embeddings(len(tokenizer))

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total_params:,} total | {trainable_params:,} trainable "
                    f"({trainable_params / total_params * 100:.2f}%)")

    num_training_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    param_groups = create_optimizer_grouped_params(model, args)

    if use_deepspeed:
        ds_config = load_deepspeed_config(args.deepspeed, args, world_size, num_training_steps, num_warmup_steps)
        model_engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=param_groups, config=ds_config)
        model_to_save = model_engine.module
    else:
        model = model.to(device)
        model_engine = model
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model_engine = DDP(model, device_ids=[local_rank], output_device=local_rank,
                               find_unused_parameters=True, static_graph=True)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        get_scheduler = get_cosine_schedule_with_warmup if args.scheduler_type == "cosine" else get_linear_schedule_with_warmup
        scheduler = get_scheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        model_to_save = model

    def save(name):
        path = os.path.join(args.output_dir, name)
        model_to_save.save_pretrained(path)
        model_to_save.save_tokenizer(tokenizer, path)

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model_engine, train_dataloader, epoch, args, device, optimizer, scheduler,
            use_deepspeed=use_deepspeed, rank=rank, use_mlflow=use_mlflow, global_step=global_step,
        )
        global_step += train_metrics["steps"]

        val_metrics = evaluate(model_engine, val_dataloader, device) if val_dataloader is not None else None

        if is_main:
            log_msg = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['token_accuracy']:.4f}"
            if val_metrics:
                log_msg += f", val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['token_accuracy']:.4f}"
                if use_mlflow:
                    log_mlflow({"val_loss": val_metrics["loss"], "val_token_accuracy": val_metrics["token_accuracy"]}, global_step)
            logger.info(log_msg)

            if args.save_every_epoch:
                save(f"checkpoint_epoch_{epoch}")
            if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save("best_model")
                logger.info(f"New best model (val_loss={best_val_loss:.4f})")

    if is_main:
        save("final_model")
        if use_mlflow:
            mlflow.end_run()

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="BrepCoder Phase 2 training (reverse engineering)")

    # Data
    parser.add_argument("--code_dir", type=str, required=True, help="Directory with the CAD code JSON files")
    parser.add_argument("--graph_dir", type=str, required=True, help="Directory with the B-rep graph .bin files")
    parser.add_argument("--split_json", type=str, required=True, help="Path to train_val_test_split.json")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum token length (prompt + code)")

    # Model
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the base LLM (Qwen2.5-1.5B)")
    parser.add_argument("--code_hidden_size", type=int, default=1536, help="Projector output size (= LLM hidden size)")
    parser.add_argument("--brep_encoder_checkpoint", type=str, default=None,
                        help="Phase 1 B-rep encoder checkpoint. If missing, the encoder is randomly initialized.")
    parser.add_argument("--graph_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_num_layers", type=int, default=3)
    parser.add_argument("--graph_num_mlp_layers", type=int, default=2)

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16, help="Micro batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--llm_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "linear"],
                        help="LR schedule without DeepSpeed (DeepSpeed uses the scheduler in its config)")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_projection", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate on the validation split after every epoch")
    parser.add_argument("--save_every_epoch", action="store_true")

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config (e.g. ds_config/zero2.json)")
    parser.add_argument("--local_rank", type=int, default=-1)

    # MLflow (optional)
    parser.add_argument("--use_mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment_name", type=str, default="BrepCoder")
    parser.add_argument("--mlflow_run_name", type=str, default=None)
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)

    train(parser.parse_args())


if __name__ == "__main__":
    main()
