import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
import dgl
from tqdm import tqdm


class CoCaDataset(Dataset):
    def __init__(self, 
                 code_dir: str, 
                 brep_dir: str, 
                 tokenizer, 
                 max_seq_len: int = 2048, 
                 split: str = "train",
                 split_json: str = None,
                 vocab_map_path: str = "./cad_vocab_map.json",
                 cache_graphs: bool = True):
        
        self.code_dir = Path(code_dir)
        self.brep_dir = Path(brep_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_graphs = cache_graphs
        self.graph_cache = {}
        
        self.samples = []
        if split_json and os.path.exists(split_json):
            with open(split_json, "r") as f:
                data = json.load(f)
                split_key = "validation" if split == "val" else split
                entry_ids = set(data.get(split_key, []))
            
            if len(entry_ids) == 0:
                print(f"Warning: No entries found for split '{split_key}' in split_json")
            
            # Load all JSON files and check entries inside
            import glob
            files = glob.glob(os.path.join(code_dir, "*.json"))
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        file_data = json.load(file)
                    
                    # Handle both dict and list formats
                    if isinstance(file_data, list):
                        entries = file_data
                    else:
                        entries = [file_data]
                    
                    # Check each entry in the file
                    for entry in entries:
                        if isinstance(entry, dict):
                            entry_id = entry.get("id", "")
                            # Check if this entry_id is in the split
                            if entry_id in entry_ids:
                                code_text = entry.get("code", "")
                                if not code_text:
                                    # Try conversations format
                                    conversations = entry.get("conversations", [])
                                    for conv in conversations:
                                        if conv.get("from") == "gpt":
                                            code_text = conv.get("value", "")
                                            break
                                
                                if code_text:
                                    # Check if graph file exists
                                    graph_subdir = self.brep_dir / entry_id[:4]
                                    graph_file = graph_subdir / f"{entry_id}.bin"
                                    
                                    if graph_file.exists():
                                        self.samples.append({
                                            "id": entry_id,
                                            "code": code_text,
                                            "graph_path": str(graph_file)
                                        })
                except Exception as e:
                    print(f"Error reading {f}: {e}")
        else:
            # No split_json, load all entries from all files
            import glob
            files = glob.glob(os.path.join(code_dir, "*.json"))
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        file_data = json.load(file)
                    
                    # Handle both dict and list formats
                    if isinstance(file_data, list):
                        entries = file_data
                    else:
                        entries = [file_data]
                    
                    for entry in entries:
                        if isinstance(entry, dict):
                            entry_id = entry.get("id", "")
                            code_text = entry.get("code", "")
                            if not code_text:
                                conversations = entry.get("conversations", [])
                                for conv in conversations:
                                    if conv.get("from") == "gpt":
                                        code_text = conv.get("value", "")
                                        break
                            
                            if code_text:
                                graph_subdir = self.brep_dir / entry_id[:4]
                                graph_file = graph_subdir / f"{entry_id}.bin"
                                
                                if graph_file.exists():
                                    self.samples.append({
                                        "id": entry_id,
                                        "code": code_text,
                                        "graph_path": str(graph_file)
                                    })
                except Exception as e:
                    print(f"Error reading {f}: {e}")


        print(f"Loading Vocab Map from {vocab_map_path}...")
        with open(vocab_map_path, 'r') as f:
            vocab_data = json.load(f)
            self.old_to_new = {int(k): v for k, v in vocab_data["old_to_new"].items()}
            self.new_vocab_size = vocab_data["vocab_size"]

        old_eos_id = self.tokenizer.eos_token_id
        if old_eos_id in self.old_to_new:
            self.new_eos_id = self.old_to_new[old_eos_id]
        else:
            # Fallback: use last token ID if eos_token not in vocab map
            self.new_eos_id = self.new_vocab_size - 1
            print(f"Warning: EOS token {old_eos_id} not in vocab map, using fallback {self.new_eos_id}")
            
        self.new_pad_id = self.new_eos_id
        
        if self.cache_graphs and len(self.samples) > 0:
            print(f"🔄 Caching {len(self.samples)} graphs to memory (this may take a while)...")
            self._preload_graphs()
            print(f"✅ Graph caching complete!")
    
    def _preload_graphs(self):
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            rank = 0
        
        iterator = tqdm(self.samples, desc="Caching graphs", disable=(rank != 0))
        
        for sample in iterator:
            graph_path = sample["graph_path"]
            if graph_path not in self.graph_cache:
                try:
                    graphs, _ = dgl.load_graphs(graph_path)
                    graph = graphs[0]
                    if "x" in graph.ndata:
                        graph.ndata["x"] = graph.ndata["x"].float()
                    if "x" in graph.edata:
                        graph.edata["x"] = graph.edata["x"].float()
                    self.graph_cache[graph_path] = graph
                except Exception as e:
                    graph = dgl.graph(([], []), num_nodes=1)
                    graph.ndata["x"] = torch.zeros(1, 7, dtype=torch.float)
                    graph.edata["x"] = torch.zeros(0, 6, dtype=torch.float)
                    self.graph_cache[graph_path] = graph
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        # code is already extracted as text in __init__
        code_text = sample["code"]

        encoded_ids = self.tokenizer.encode(
            code_text,
            truncation=True,
            max_length=self.max_seq_len - 1, 
            add_special_tokens=False
        )
        
        new_ids = [self.old_to_new.get(tid, self.new_pad_id) for tid in encoded_ids]
        
        input_ids = torch.tensor(new_ids + [self.new_eos_id], dtype=torch.long)
        
        current_len = input_ids.size(0)
        pad_len = self.max_seq_len - current_len
        
        if pad_len > 0:
            padding = torch.full((pad_len,), self.new_pad_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=0)
            
            attention_mask = torch.cat([
                torch.ones(current_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ], dim=0)
        else:
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)

        graph_path = sample["graph_path"]
        if self.cache_graphs and graph_path in self.graph_cache:
            graph = self.graph_cache[graph_path]
        else:
            try:
                graphs, _ = dgl.load_graphs(graph_path)
                graph = graphs[0]
                if "x" in graph.ndata:
                    graph.ndata["x"] = graph.ndata["x"].float()
                if "x" in graph.edata:
                    graph.edata["x"] = graph.edata["x"].float()
            except Exception as e:
                graph = dgl.graph(([], []), num_nodes=1)
                graph.ndata["x"] = torch.zeros(1, 7, dtype=torch.float)
                graph.edata["x"] = torch.zeros(0, 6, dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph": graph,
            "id": str(sample["id"]),
            "code": code_text,
        }

def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    max_len_in_batch = attention_mask.sum(dim=1).max().item()

    input_ids = input_ids[:, :max_len_in_batch]
    attention_mask = attention_mask[:, :max_len_in_batch]

    graphs = [item["graph"] for item in batch]
    ids = [item["id"] for item in batch]
    codes = [item["code"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graphs": graphs,
        "ids": ids,
        "codes": codes,
    }
