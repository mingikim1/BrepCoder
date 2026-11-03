"""
Phase 2 inference: generate Python-like CAD code from B-rep graphs.

Usually launched through `scripts/inference.sh`. Direct usage:
    # Evaluate a split -> {"results": [{"id", "original_code", "generated_code"}, ...]}
    python inference.py --model_path output/reverse_engineering/best_model --llm_model_path ... \
        --code_dir ... --graph_dir ... --split_json ... --split test

    # Single B-rep graph
    python inference.py --model_path ... --llm_model_path ... --input_graph path/to/00000007.bin
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from constants import DEFAULT_BREP_END_TOKEN, DEFAULT_BREP_START_TOKEN, DEFAULT_BREP_TOKEN
from dataset import BrepCodeDataset, load_graph, prepare_tokenizer, tokenize_sample
from model import BrepCoderModel


def load_tokenizer(model_path, llm_model_path):
    """Use the tokenizer saved with the checkpoint; otherwise rebuild it from the base LLM."""
    saved_tokenizer_path = os.path.join(model_path, "tokenizer")
    if os.path.exists(saved_tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(saved_tokenizer_path, trust_remote_code=True)
        print(f"Loaded tokenizer from {saved_tokenizer_path} (vocab size: {len(tokenizer)})")
        return tokenizer
    return prepare_tokenizer(llm_model_path)


def build_generate_kwargs(tokenizer, args):
    do_sample = args.temperature > 0 if args.do_sample == "auto" else args.do_sample == "true"
    if args.num_beams > 1 and args.do_sample == "auto":
        do_sample = False

    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "num_beams": args.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.num_beams > 1:
        kwargs["early_stopping"] = True
    if do_sample:
        kwargs["temperature"] = args.temperature if args.temperature > 0 else 1.0
        kwargs["top_p"] = args.top_p
    if args.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = args.repetition_penalty
    return kwargs


def decode_generated(token_ids, tokenizer):
    """Cut the generated ids at EOS and strip the special tokens from the decoded text."""
    if token_ids and token_ids[0] in (tokenizer.eos_token_id, tokenizer.bos_token_id):
        token_ids = token_ids[1:]
    if tokenizer.eos_token_id in token_ids:
        token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id)]

    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for token in [DEFAULT_BREP_TOKEN, DEFAULT_BREP_START_TOKEN, DEFAULT_BREP_END_TOKEN,
                  tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token]:
        if token:
            text = text.replace(token, "")
    return text


@torch.no_grad()
def generate_code(model, tokenizer, graph_path, device, generate_kwargs):
    """Generate CAD code for one B-rep graph."""
    input_ids, _ = tokenize_sample(tokenizer)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    graph = load_graph(graph_path).to(device)

    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, graph=graph, **generate_kwargs)
    return decode_generated(output_ids[0].tolist(), tokenizer)


def save_json(data, path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="BrepCoder Phase 2 inference (reverse engineering)")

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint saved by train.py (e.g. .../best_model)")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the base LLM (Qwen2.5-1.5B)")
    parser.add_argument("--code_hidden_size", type=int, default=1536)
    parser.add_argument("--graph_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_num_layers", type=int, default=3)
    parser.add_argument("--graph_num_mlp_layers", type=int, default=2)

    # Input: a whole split, or a single graph
    parser.add_argument("--code_dir", type=str, default=None)
    parser.add_argument("--graph_dir", type=str, default=None)
    parser.add_argument("--split_json", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate only the first N samples of the split")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the samples into N contiguous shards (multi-GPU)")
    parser.add_argument("--shard_id", type=int, default=0, help="Index of the shard processed by this run")
    parser.add_argument("--input_graph", type=str, default=None, help="Single B-rep graph (.bin) to run inference on")
    parser.add_argument("--output_file", type=str, default=None)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 for greedy decoding")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=str, default="auto", choices=["auto", "true", "false"])

    # Device
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")

    args = parser.parse_args()
    if not args.input_graph and not (args.code_dir and args.graph_dir and args.split_json):
        parser.error("Provide --input_graph, or --code_dir, --graph_dir and --split_json")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(args.model_path, args.llm_model_path)
    model = BrepCoderModel.from_pretrained(
        model_path=args.model_path,
        llm_model_path=args.llm_model_path,
        code_hidden_size=args.code_hidden_size,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_num_layers=args.graph_num_layers,
        graph_num_mlp_layers=args.graph_num_mlp_layers,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_flash_attn=args.use_flash_attn,
    )
    model.llm.config.pad_token_id = tokenizer.pad_token_id
    if len(tokenizer) > model.llm.config.vocab_size:
        model.llm.resize_token_embeddings(len(tokenizer))
    model = model.to(device).eval()
    generate_kwargs = build_generate_kwargs(tokenizer, args)

    # Single graph
    if args.input_graph:
        generated = generate_code(model, tokenizer, args.input_graph, device, generate_kwargs)
        print("=" * 60)
        print(generated)
        print("=" * 60)
        output_file = args.output_file or os.path.join(args.model_path, "inference_result.json")
        save_json({"input_graph": args.input_graph, "generated_code": generated}, output_file)
        return

    # Whole split
    dataset = BrepCodeDataset(
        code_dir=args.code_dir, graph_dir=args.graph_dir, split_json=args.split_json,
        tokenizer=tokenizer, split=args.split,
    )
    indices = list(range(len(dataset)))
    if args.max_samples:
        indices = indices[:args.max_samples]
    shard_size = (len(indices) + args.num_shards - 1) // args.num_shards
    indices = indices[args.shard_id * shard_size:(args.shard_id + 1) * shard_size]

    results = []
    for idx in tqdm(indices, desc=f"Shard {args.shard_id}/{args.num_shards}"):
        sample = dataset.samples[idx]
        generated = generate_code(model, tokenizer, sample["graph_path"], device, generate_kwargs)
        results.append({"id": sample["id"], "original_code": sample["code"], "generated_code": generated})

    output_file = args.output_file or os.path.join(args.model_path, f"inference_results_{args.split}.json")
    save_json({"results": results}, output_file)


if __name__ == "__main__":
    main()
