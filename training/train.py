import pytorch_lightning as pl
from model.lightning_model import SymmetryTransformer
from pytorch_lightning.callbacks import EarlyStopping
from configs.config import LEARNING_RATE, HIDDEN_DIM, NUM_PREDS, EARLY_STOP_PATIENCE, MAX_EPOCHS, RANDOM_STATE, NUM_LAYERS, NUM_HEADS, DATASET_PATH, NO_ATTN
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

"""
To run training:
    python3 train.py --backbone pointnetpp/pointnet/None
"""

# Tell PyTorch & cuDNN to pick only deterministic kernels
pl.seed_everything(RANDOM_STATE, workers=True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LossPlotter(pl.Callback):
    """Callback to store losses for plotting after training."""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.item())


def _draw_lines(ax, lines, style, assume="cos_sin_and_d"):
    """
    lines:
      [K,3] -> (cosθ, sinθ, d) if assume == 'cos_sin_and_d'   with line n·x + d = 0
      [K,3] -> (nx,  ny,  d)   if assume == 'normal_and_d'    (n will be normalized)
    """
    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu().numpy()
    if lines is None or (hasattr(lines, "size") and lines.size == 0):
        return

    # ensure axes are initialized (scatter should have set them)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = 1.5 * max(x1 - x0, y1 - y0)  # half-length, long enough to span the plot

    for row in np.asarray(lines):
        if row is None or len(row) == 0:
            continue

        if assume == "cos_sin_and_d":
            nx, ny, d = float(row[0]), float(row[1]), float(row[2])
            n = np.array([nx, ny], dtype=float)
        elif assume == "normal_and_d":
            nx, ny, d = float(row[0]), float(row[1]), float(row[2])
            n = np.array([nx, ny], dtype=float)
        else:
            raise ValueError(f"Unknown assume='{assume}'")

        # normalize the normal (safety)
        norm = np.linalg.norm(n)
        print("norm:", norm)
        if norm < 1e-12:
            continue
        n = n / norm

        # direction along line (perpendicular to n)
        t = np.array([-n[1], n[0]], dtype=float)

        # pick a point on the line:
        # n·x + d = 0  =>  n·x = -d  =>  x0 = -d * n  (since n is unit)
        p0 = d * n

        P = np.stack([p0 - L * t, p0 + L * t], axis=0)
        ax.plot(P[:, 0], P[:, 1], style, linewidth=1.5)


def visualize_predictions(model, dataset, num_samples=5):
    """Visualize model predictions against ground truth (both as (cosθ, sinθ, d))."""
    model.eval()
    dataset.set_get_points_mode(True)  # enable returning raw points if available
    indxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    for i in indxs:
        points, features, targets= dataset[i]
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], s=5)

        # targets/preds now match (cosθ, sinθ, d)
        _draw_lines(ax, targets, "r-", assume="cos_sin_and_d")
        with torch.inference_mode():
            print(model.backbone_model)
            if model.backbone_model == "pointnetpp":
                pred_lines = model(features[None, ...].float().to(model.device))[0]  # [num_preds, 4]
            else:
                pred_lines = model(points[None, ...].float().to(model.device))[0]  # [num_preds, 4]
        print(pred_lines.shape)
        print(pred_lines)
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
        print("targets:", targets, "predictions:", pred_lines)
        plt.savefig(f"prediction_{i}.png")
        plt.close(fig)


def plot_losses(callback: LossPlotter):
    """Plot training and validation loss curves."""
    plt.figure()
    plt.plot(callback.train_losses, label="Train Loss")
    plt.plot(callback.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("loss_curve.png")


def main():
    print("PY:", sys.executable)
    print("TORCH:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        # force a tiny CUDA kernel right here
        torch.cuda.synchronize()
        x = torch.randn(4, device="cuda")
        x = x.sin_()  # simple kernel
        torch.cuda.synchronize()
        print("Tiny CUDA op: OK")
        
    dataste_path = "pointNetpp_embeddings.npz"
    print(f"Using dataset: {dataste_path}")

    # Create model
    model = SymmetryTransformer(
        hidden_dim=HIDDEN_DIM,
        nheads=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        input_dim=2,
        num_preds=NUM_PREDS,
        learning_rate=LEARNING_RATE,
        output_dim=4,
        dataset_path=dataste_path,
        backbone_model=args.backbone,
        no_attn=NO_ATTN
    )
    
    loss_callback = LossPlotter()
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="train_loss",      # metric to monitor
        patience=EARLY_STOP_PATIENCE,  # epochs to wait for improvement
        verbose=False,
        mode="min"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Use GPU if available
        devices="auto",
        log_every_n_steps=3,
        enable_checkpointing=True,
        callbacks=[
            loss_callback, 
            early_stop_callback,
        ],
    )

    # Train the model
    trainer.fit(model)
    # Test the model
    trainer.test(model)

    # Plot losses and visualize predictions
    plot_losses(loss_callback)
    visualize_predictions(model, model.test_dataloader().dataset.dataset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="pointnetpp", choices=["None", "pointnet", "pointnetpp"], help="Which backbone to use")
    args = ap.parse_args()
    main()
