# SymTr - Symmetry Transformer

SymTr is a codebase for learning symmetry lines from 2D point clouds with a DETR-like transformer. The model combines optional PointNet/PointNet++ backbones with a transformer encoder-decoder that predicts line parameters $(\cos\theta, \sin\theta, d)$ and a confidence score for each candidate symmetry. Training, evaluation, and analysis are built on PyTorch Lightning.

## Highlights
- DETR-inspired architecture (`model/SymTr.py`) tailored to symmetry-line detection in normalized 2D point clouds.
- Switchable backbones: raw coordinates, PointNet, or a frozen PointNet++ embedding (`--backbone` flag).
- Synthetic dataset generator (`data/generate_dataset_pytorch.py`) and feature extraction pipeline (`data/extract_features_numpy.py`).
- Deterministic Lightning training loop (`training/train.py`) with early stopping, plotting helpers, and prediction snapshots.
- Experiment tooling: Optuna hyperparameter search (`training/tune_model.py`), evaluation metrics (`utils/calculate_metrics.py`), and attention visualizations (`utils/attention_maps_standalone.py`).
- We provide a small 100 samples dataset: dataset.npz and extracted pointNet++ features version dataset: pointNet_embeddings.npz.
- We also provide a pre-trained SymTr model with pointNet++ backbone in trained_model_pointnetpp.

## Repository Layout
- `configs/` – Default training and tuning hyperparameters.
- `data/` – Scripts for synthesizing datasets and extracting frozen PointNet++ features.
- `loss/` – Hungarian matcher and DETR-style loss (`SetCriterion`).
- `model/` – Symmetry transformer, Lightning wrapper, and backbone definitions.
- `training/` – Entry points for training and hyperparameter optimisation.
- `utils/` – Metric computation and attention-map tooling.

## Quick Start
### 1. Create an environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Install a PyTorch build that matches your CUDA setup before `pip install -r requirements.txt` if you need GPU acceleration.

### 2. Prepare data
```bash
# Generate a synthetic dataset of point clouds + symmetry labels
python data/generate_dataset_pytorch.py --num_samples 8000

# Convert to the dense NPZ format expected by SymmetryDataset
python data/extract_features_numpy.py
```

### 3. Train a model
```bash
# Train with PointNet++ features (default)
python training/train.py --backbone pointnetpp

# Alternatives:
# python training/train.py --backbone pointnet   # learn features end-to-end
# python training/train.py --backbone None        # transformer on raw coordinates
```
Lightning checkpoints and logs are written under `lightning_logs/`, while the helper callback saves `loss_curve.png` and `prediction_*.png` in the project root.

### 4. Evaluate a checkpoint
```bash
python utils/calculate_metrics.py --ckpt lightning_logs/version_0/checkpoints/epoch=*.ckpt
```
The script reports the mean direction and offset errors on the model test set.
In addition, for visualizing model predictions, can use inference.py

## Configuration
Hyperparameters such as learning rate, transformer depth, loss weights, and dataset location configured in `configs/config.py`.

## Hyperparameter Search
`training/tune_model.py` wraps Optuna with K-fold cross-validation. Results are stored in .sqlite output file.
```bash
python training/tune_model.py --backbone pointnetpp
```
Adjust the search space inside `objective()` before launching longer studies.

## Visualization Utilities
- `utils/attention_maps_standalone.py` captures encoder/decoder attention maps
Example:
  ```bash
  python utils/attention_maps_standalone.py --ckpt path/to/epoch.ckpt --out_dir attn_viz
  ```
- `inference.py` generate qualitative overlays of predicted vs. ground-truth symmetry lines via `visualize_predictions()`.
