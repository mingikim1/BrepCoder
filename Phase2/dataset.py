"""
Dataset for B-rep reverse engineering: B-rep graph -> Python-like CAD code.

Expected data layout:
    code_dir/*.json                 each file holds a list of {"id": "00000007", "code": "..."}
    graph_dir/<id[:4]>/<id>.bin     DGL face-adjacency graph of the B-rep with that id
    split_json                      {"train": [ids], "validation": [ids], "test": [ids]}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dgl
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from constants import (
    BREP_PROMPT,
    BREP_TOKEN_INDEX,
    DEFAULT_BREP_END_TOKEN,
    DEFAULT_BREP_START_TOKEN,
    DEFAULT_BREP_TOKEN,
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
)


def load_graph(graph_path: str) -> dgl.DGLGraph:
    """
    Load a B-rep graph saved with `dgl.save_graphs`.

    ndata["x"]: [num_faces, 10, 10, 7] UV-grid samples of each surface
    edata["x"]: [num_edges, 10, 6]     U-grid samples of each curve
    """
    try:
        graph = dgl.load_graphs(graph_path)[0][0]
        if "x" in graph.ndata:
            graph.ndata["x"] = graph.ndata["x"].float()
        if "x" in graph.edata:
            graph.edata["x"] = graph.edata["x"].float()
    except Exception as e:
        # Fall back to a single-node placeholder so that one unreadable file does not stop a run.
        print(f"Error loading graph {graph_path}: {e}")
        graph = dgl.graph(([], []), num_nodes=1)
        graph.ndata["x"] = torch.randn(1, 10, 10, 7)
        graph.edata["x"] = torch.randn(0, 10, 6)
    return graph


def tokenize_sample(tokenizer, code: Optional[str] = None) -> Tuple[List[int], List[int]]:
    """
    Tokenize the prompt `<brep_start><brep><brep_end>` followed by `code<eos>` as the target.

    The <brep> token id is replaced by BREP_TOKEN_INDEX, which the model later swaps for the
    B-rep node embeddings. Prompt positions are masked out of the labels. With `code=None`,
    only the prompt is returned (used for generation).
    """
    prompt_ids = tokenizer(BREP_PROMPT, add_special_tokens=True)["input_ids"]
    target_ids = []
    if code is not None:
        target_ids = tokenizer(f"{code}{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]

    brep_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BREP_TOKEN)
    input_ids = [BREP_TOKEN_INDEX if t == brep_token_id else t for t in prompt_ids + target_ids]
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return input_ids, labels


class BrepCodeDataset(Dataset):
    """Pairs each B-rep graph with its Python-like CAD code for one split."""

    def __init__(
        self,
        code_dir: str,
        graph_dir: str,
        split_json: str,
        tokenizer,
        split: str = "train",
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.code_dir = Path(code_dir)
        self.graph_dir = Path(graph_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length

        self.samples = self._load_samples(split_json)
        print(f"Loaded {len(self.samples)} samples for {split}")

    def _load_samples(self, split_json: str) -> List[Dict[str, Any]]:
        split_key = "validation" if self.split == "val" else self.split
        with open(split_json, "r") as f:
            split_data = json.load(f)
        if split_key not in split_data:
            raise ValueError(f"Split '{split_key}' not found in {split_json}. Available keys: {list(split_data.keys())}")
        split_ids = set(split_data[split_key])

        samples = []
        skipped = 0
        for json_file in sorted(self.code_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if isinstance(entries, dict):
                entries = [entries]

            for entry in entries:
                entry_id = entry.get("id", "")
                if entry_id not in split_ids:
                    continue
                code = entry.get("code", "")
                graph_path = self.graph_dir / entry_id[:4] / f"{entry_id}.bin"
                if not code or not graph_path.exists():
                    skipped += 1
                    continue
                samples.append({"id": entry_id, "code": code, "graph_path": str(graph_path)})

        if skipped > 0:
            print(f"  Skipped {skipped} '{split_key}' samples with missing code or graph file")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        input_ids, labels = tokenize_sample(self.tokenizer, sample["code"])
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "graph": load_graph(sample["graph_path"]),
            "id": sample["id"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Left-pad the token sequences and batch the DGL graphs."""
    batch_size = len(batch)
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, -seq_len:] = item["input_ids"]
        attention_mask[i, -seq_len:] = item["attention_mask"]
        labels[i, -seq_len:] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "graph": dgl.batch([item["graph"].cpu() for item in batch]),
        "ids": [item["id"] for item in batch],
    }


def prepare_tokenizer(tokenizer_path: str):
    """Load the LLM tokenizer and add the B-rep special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({
        "additional_special_tokens": [DEFAULT_BREP_START_TOKEN, DEFAULT_BREP_TOKEN, DEFAULT_BREP_END_TOKEN],
    })
    tokenizer.padding_side = "left"

    print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {len(tokenizer)})")
    return tokenizer
