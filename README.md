# MAI 645 Final Project - Character Motion Prediction

This repository contains the MAI 645 final project scaffold for comparing three character motion representations for motion prediction:

1. Positional representation
2. Euler angle representation
3. Quaternion representation

The assignment handout names the required scripts, but the original GitHub repository link and BVH dataset path are still unknown. Those locations are marked with `TODO` placeholders in the notebook and command examples below.

## Repository Layout

```text
code/
  generate_training_pos_data.py
  generate_training_euler_data.py
  generate_training_quad_data.py
  pytorch_train_pos_aclstm.py
  pytorch_train_euler_aclstm.py
  pytorch_train_quad_aclstm.py
  synthise_pos_motion.py
  synthise_euler_motion.py
  synthise_quad_motion.py
  mai645_runner.py
notebooks/
  MAI645_colab_runner.ipynb
scripts/
  setup_colab.sh
results/
  plots/
  tables/
```

## Data Policy

Do not commit heavy project files. Keep BVH data, processed arrays, model checkpoints, and generated outputs in Google Drive or another external storage location.

The repository ignores:

- `data/`, `processed/`, `models/`, `outputs/`
- `*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `*.bvh`
- common run logs and archive/checkpoint formats

## Google Colab Workflow

Open `notebooks/MAI645_colab_runner.ipynb` in Colab and update the TODO path variables:

```python
PROJECT_ROOT = "/content/drive/MyDrive/MAI645/mai645-character-motion-prediction"  # TODO
ASSIGNMENT_REPO_PATH = "/content/drive/MyDrive/MAI645/TODO_original_assignment_repo"  # TODO
BVH_DATASET_PATH = "/content/drive/MyDrive/MAI645/TODO_bvh_dataset_folder"  # TODO
DRIVE_RUN_ROOT = "/content/drive/MyDrive/MAI645/runs"  # TODO
```

The notebook mounts Google Drive, installs `requirements.txt`, and executes the required preprocessing, training, and evaluation entrypoints. It defaults to `DRY_RUN = True` so the full Colab flow can be checked before the missing assignment repository and BVH dataset are added. Set `DRY_RUN = False` after replacing the TODO paths and filling in the adapter logic in `code/mai645_runner.py`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Colab or a fresh runtime:

```bash
bash scripts/setup_colab.sh
```

## Required Assignment Commands

### Preprocessing

```bash
python code/generate_training_pos_data.py \
  --assignment-repo TODO_original_assignment_repo \
  --bvh-dir TODO_bvh_dataset_folder \
  --output-dir processed/pos

python code/generate_training_euler_data.py \
  --assignment-repo TODO_original_assignment_repo \
  --bvh-dir TODO_bvh_dataset_folder \
  --output-dir processed/euler

python code/generate_training_quad_data.py \
  --assignment-repo TODO_original_assignment_repo \
  --bvh-dir TODO_bvh_dataset_folder \
  --output-dir processed/quad
```

### Training

```bash
python code/pytorch_train_pos_aclstm.py --processed-dir processed/pos --model-dir models/pos
python code/pytorch_train_euler_aclstm.py --processed-dir processed/euler --model-dir models/euler
python code/pytorch_train_quad_aclstm.py --processed-dir processed/quad --model-dir models/quad
```

### Evaluation and BVH Synthesis

```bash
python code/synthise_pos_motion.py --processed-dir processed/pos --model-path models/pos/TODO_model.pt --output-dir outputs/pos
python code/synthise_euler_motion.py --processed-dir processed/euler --model-path models/euler/TODO_model.pt --output-dir outputs/euler
python code/synthise_quad_motion.py --processed-dir processed/quad --model-path models/quad/TODO_model.pt --output-dir outputs/quad
```

## Implementation Notes

`code/mai645_runner.py` centralizes argument parsing and TODO validation for the nine required scripts. Replace the TODO adapter sections there once the original assignment repository and BVH dataset layout are known.
