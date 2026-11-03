# build_vocab.py
import json
import torch
import glob
import os
from tqdm import tqdm
from transformers import AutoTokenizer

def build_vocab():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # BrepCoder/

    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Checkpoint", "Qwen2.5-1.5B")
    DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset", "code")
    SAVE_PATH = os.path.join(SCRIPT_DIR, "cad_vocab_map.json")
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    
    special_tokens = {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    }
    
    used_tokens = set(special_tokens)
    
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    print(f"Scanning {len(files)} files...")
    
    for fpath in tqdm(files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            # Handle both dict and list formats
            if isinstance(data, list):
                # If it's a list, iterate through entries
                for entry in data:
                    if isinstance(entry, dict):
                        code_text = entry.get("code", "")
                        if code_text:
                            ids = tokenizer.encode(code_text, add_special_tokens=False)
                            used_tokens.update(ids)
            elif isinstance(data, dict):
                # If it's a dict, get code directly
                code_text = data.get("code", "")
                if code_text:
                    ids = tokenizer.encode(code_text, add_special_tokens=False)
                    used_tokens.update(ids)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    sorted_tokens = sorted(list(used_tokens))
    
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_tokens)}
    new_to_old = {new_id: old_id for new_id, old_id in enumerate(sorted_tokens)}
    
    print(f"Original Vocab Size: {len(tokenizer)}")
    print(f"Reduced Vocab Size: {len(old_to_new)}")
    
    with open(SAVE_PATH, "w") as f:
        json.dump({
            "old_to_new": old_to_new, 
            "new_to_old": new_to_old,
            "vocab_size": len(old_to_new)
        }, f)
        
    print(f"Saved vocab map to {SAVE_PATH}")

if __name__ == "__main__":
    build_vocab()