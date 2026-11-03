# BrepCoder: A Unified Multimodal Large Language Model for Multi-task B-rep Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2602.22284-b31b1b.svg)](https://arxiv.org/abs/2602.22284)

Official implementation of **BrepCoder** (ECCV 2026).

## 🔧 Installation

Tested with Python 3.10 and CUDA 12.1.

```bash
conda create -n brepcoder python=3.10 -y
conda activate brepcoder

# PyTorch 2.1.0 (CUDA 12.1)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# DGL 1.1.3 (CUDA 12.1)
pip install dgl==1.1.3+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
# Other dependencies
pip install -r requirements.txt
# FlashAttention-2 (optional, set USE_FLASH_ATTN=false in the scripts to run without it)
pip install flash-attn==2.3.2 --no-build-isolation
```

## 📦 Data Preparation

### Dataset

We convert the [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) dataset into the Python-like CAD code format and pair each model with its B-rep graph. Download the following files from [Google Drive](https://drive.google.com/drive/folders/1v723gHq9k2xCv9zABIkazESEzQMafy4_?usp=sharing) and place them in `Dataset/`:

| File | Description |
| :--- | :--- |
| `code.tar.gz` | Python-like CAD code (`code/0000.json` – `code/0099.json`) |
| `graph.tar.gz` | B-rep graphs (`graph/<first 4 digits of id>/<id>.bin`) |
| `train_val_test_split.json` | Train / validation / test split (90:5:5) |

Then extract the archives:

```bash
cd Dataset
tar -xzf code.tar.gz
tar -xzf graph.tar.gz
cd ..
```

- Each `code/*.json` file holds a list of `{"id": "00000007", "code": "sketch0 = [] ..."}` entries.
- Each graph is a [DGL](https://www.dgl.ai/) face-adjacency graph following [UV-Net](https://github.com/AutodeskAILab/UV-Net): nodes are faces with UV-grid samples (`ndata["x"]`: `[num_faces, 10, 10, 7]`), and edges are curves with U-grid samples (`edata["x"]`: `[num_edges, 10, 6]`).
- `train_val_test_split.json` maps `train`, `validation`, and `test` to lists of ids.

### Base LLM

Download [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) into `Checkpoint/Qwen2.5-1.5B`:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir Checkpoint/Qwen2.5-1.5B
```

The final directory layout should look like this:

```
BrepCoder/
├── Checkpoint/
│   └── Qwen2.5-1.5B/
├── Dataset/
│   ├── code/
│   │   ├── 0000.json
│   │   └── ...
│   ├── graph/
│   │   ├── 0000/
│   │   │   ├── 00000007.bin
│   │   │   └── ...
│   │   └── ...
│   └── train_val_test_split.json
├── Phase1/            # Multimodal alignment (B-rep encoder pre-training)
└── Phase2/            # Multimodal integration (B-rep encoder + projector + LLM)
```

## 🚀 Training

### Phase 1: Multimodal Alignment

```bash
bash Phase1/script/train.sh
```

This pre-trains the B-rep encoder by aligning B-rep graphs with CAD code (CoCa). By default, it runs with DeepSpeed on GPUs `0,1,2,3` (edit `--include` in the script to change this). The encoder used by Phase 2 is saved to `Phase1/output/best_model/brep_encoder.pth`.

### Phase 2: Reverse Engineering

```bash
bash Phase2/scripts/train.sh
```

This trains BrepCoder to generate Python-like CAD code from B-rep graphs. The default configuration is DeepSpeed ZeRO-2 on 4 GPUs, a batch size of 16 per GPU, 3 epochs, bf16, and FlashAttention-2. The B-rep encoder is frozen, while the projector and the LLM are trained.

- `--num_gpus`, `--epochs`, `--batch_size`, `--output_dir` and `--master_port` can be passed on the command line. Other settings (learning rates, freezing, paths, ...) are at the top of the script.
- The Phase 1 encoder is loaded from `Phase1/output/best_model/brep_encoder.pth`. If the file does not exist, the encoder is randomly initialized. To train the encoder together with the other modules, set `FREEZE_ENCODER=false`.
- Optional MLflow logging can be enabled with `USE_MLFLOW=true` (requires `pip install mlflow`).

Checkpoints are saved to `Phase2/output/reverse_engineering/`:

```
reverse_engineering/
├── best_model/
│   ├── brep_encoder/
│   ├── projection/
│   ├── llm/
│   ├── tokenizer/
│   └── config.json
├── checkpoint_epoch_N/
└── final_model/
```

## 🧪 Inference

```bash
# Test split on a single GPU
bash Phase2/scripts/inference.sh

# Test split, sharded over 4 GPUs
bash Phase2/scripts/inference.sh --num_gpus 4

# A single B-rep graph
bash Phase2/scripts/inference.sh --input_graph Dataset/graph/0000/00000007.bin
```

By default, the script loads `Phase2/output/reverse_engineering/best_model` (change it with `--model_path`) and decodes greedily. Results are written to `<model_path>/inference_results_test.json`:

```json
{
  "results": [
    {
      "id": "00000348",
      "original_code": "sketch0 = [] ...",
      "generated_code": "sketch0 = [] ..."
    }
  ]
}
```

Other options: `--split {train,val,test}`, `--max_samples N`, `--output_file FILE`, `--max_new_tokens N`, `--temperature T`.

## 📏 Evaluation

We evaluate with the [DeepCAD evaluation scripts](https://github.com/rundiwu/DeepCAD/tree/master/evaluation). First, convert the generated code into DeepCAD's command sequence format with `code_to_h5()` in [`Phase2/code_to_h5.py`](Phase2/code_to_h5.py), and save one `<id>.h5` file per sample with the prediction as `out_vec` and the ground truth as `gt_vec`:

```python
# Run from Phase2/ (requires `pip install h5py`)
import h5py
from code_to_h5 import code_to_h5

# `result` is one entry of "results" in inference_results_test.json
out_vec = code_to_h5(result["generated_code"])  # [seq_len, 17]
gt_vec = code_to_h5(result["original_code"])
with h5py.File(f"test_h5/{result['id']}.h5", "w") as fp:
    fp.create_dataset("out_vec", data=out_vec)
    fp.create_dataset("gt_vec", data=gt_vec)
```

Then run the DeepCAD evaluation on the h5 directory:

```bash
cd DeepCAD/evaluation
# Command accuracy and parameter accuracy
python evaluate_ae_acc.py --src /path/to/test_h5
# Chamfer distance and invalid ratio
python evaluate_ae_cd.py --src /path/to/test_h5 --parallel
```

`evaluate_ae_cd.py` needs DeepCAD's ground-truth point clouds. See the [DeepCAD repository](https://github.com/rundiwu/DeepCAD) for its setup.

## 🔗 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{BrepCoder_ECCV2026,
  author    = {Kim, Mingi and Kim, Yongjun and Kang, Jungwoo and Kim, Hyungki},
  title     = {{BrepCoder}: A Unified Multimodal Large Language Model for Multi-task {B}-rep Reasoning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {428--445},
  year      = {2026}
}
```

## 👏 Acknowledgements

This project builds on [UV-Net](https://github.com/AutodeskAILab/UV-Net) (B-rep encoder), [CoCa](https://arxiv.org/abs/2205.01917) (multimodal alignment), [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) (dataset), CAD-Llama (Python-like CAD code format), and [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B) (LLM backbone). We thank the authors for releasing their work. Qwen2.5-1.5B is released under the Apache-2.0 license.
