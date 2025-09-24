#!/usr/bin/env python3
"""
Inference script for SymmetryTransformer.
Loads a trained checkpoint and generates prediction visualizations.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from model.lightning_model import SymmetryTransformer
from configs.config import (
    HIDDEN_DIM, NUM_PREDS, LEARNING_RATE, NUM_LAYERS, NUM_HEADS, NO_ATTN
)

# deterministic setup
pl.seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _draw_lines(ax, lines, style, assume="cos_sin_and_d"):
    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu().numpy()
    if lines is None or (hasattr(lines, "size") and lines.size == 0):
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = 1.5 * max(x1 - x0, y1 - y0)

    for row in np.asarray(lines):
        if row is None or len(row) == 0:
            continue

        nx, ny, d = float(row[0]), float(row[1]), float(row[2])
        n = np.array([nx, ny], dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        t = np.array([-n[1], n[0]], dtype=float)
        p0 = d * n
        P = np.stack([p0 - L * t, p0 + L * t], axis=0)
        ax.plot(P[:, 0], P[:, 1], style, linewidth=1.5)


def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    dataset.set_get_points_mode(True)

    indxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    for i in indxs:
        points, features, targets = dataset[i]
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], s=5)

        _draw_lines(ax, targets, "r-", assume="cos_sin_and_d")

        with torch.inference_mode():
            if model.backbone_model == "pointnetpp":
                pred_lines = model(features[None, ...].float().to(model.device))[0]
            else:
                pred_lines = model(points[None, ...].float().to(model.device))[0]

        _draw_lines(ax, pred_lines[0], "g--", assume="cos_sin_and_d")

        ax.set_aspect("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend(handles=[
            plt.Line2D([0], [0], color="blue", marker="o", linestyle="", label="points"),
            plt.Line2D([0], [0], color="red", linestyle="-", label="targets"),
            plt.Line2D([0], [0], color="green", linestyle="--", label="predictions"),
        ])
        ax.set_title(f"Sample {i}")
        if not os.path.exists("inference_out"):
            os.makedirs("inference_out")
        plt.savefig(f"inference_out/inference_prediction_{i}.png")
        plt.close(fig)


def main(args):
    dataset_path = "pointNetpp_embeddings.npz"

    print(f"Loading model from checkpoint: {args.ckpt}")

    model = SymmetryTransformer.load_from_checkpoint(
        args.ckpt)

    trainer = pl.Trainer(accelerator="auto", devices="auto")
    trainer.test(model)

    visualize_predictions(model, model.test_dataloader().dataset.dataset, num_samples=args.num_samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt)")
    ap.add_argument("--num-samples", type=int, default=15, help="Number of random samples to visualize")
    args = ap.parse_args()
    main(args)