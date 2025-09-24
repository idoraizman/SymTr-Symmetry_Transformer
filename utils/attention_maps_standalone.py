#!/usr/bin/env python3
"""
Standalone per-sample attention visualization for a DETR-like SymmetryTransformer.

- Uses your attention capture utilities from attention_tools.py
  (capture_any_transformer_attn, _to_BHTS, etc.).
- Saves ONE PNG per (sample, query_index) that contains:
    * the input points
    * the predicted symmetry line from the SAME query you're visualizing
    * a grid: ALL layers × ALL heads attention maps for the chosen bucket
- Handles (cosθ, sinθ, d) line parameterization with n·x + d = 0.
- Works headless (Agg), suitable for SSH. No callbacks / Lightning hooks required.
"""

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model.lightning_model import SymmetryTransformer
from model.pointNetpp_pretrained import get_model
from data.symmetry_dataset import SymmetryDataset
from utils.attention import capture_any_transformer_attn

# ---------------------- Utilities ----------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def draw_lines_cos_sin_d(ax, lines, style="g--"):
    """
    Draw lines parameterized by (cosθ, sinθ, d) with equation: n·x + d = 0, where n = (cosθ, sinθ).
    'lines' can be [K,3] or [3].
    """
    if lines is None:
        return
    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu().numpy()
    arr = np.asarray(lines)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0:
        return

    x0, x1 = -1.0, 1.0
    y0, y1 = -1.0, 1.0
    L = 1.5 * max(x1 - x0, y1 - y0)

    for row in arr:
        nx, ny, d = float(row[0]), float(row[1]), float(row[2])
        n = np.array([nx, ny], dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        t = np.array([-n[1], n[0]], dtype=float)  # tangent direction
        p0 = -d * n                                # any point on the line
        P = np.stack([p0 - L * t, p0 + L * t], axis=0)
        ax.plot(P[:, 0], P[:, 1], style, linewidth=1.6)

def infer_outputs(model, points_b1: torch.Tensor):
    """
    Expect model(points) -> (pred_lines, confidence)
      pred_lines: [1, K, 3] (cos, sin, d)
      confidence: [1, K]    (logits or probs)
    Returns dict with pred_lines [K,3], confidence [K]
    """
    out = model(points_b1)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        pred_lines, confidence = out[0], out[1]
    elif isinstance(out, dict):
        pred_lines = out.get("pred_lines", None)
        confidence = out.get("confidence", out.get("pred_logits", None))
        if pred_lines is None or confidence is None:
            raise KeyError("Could not find ('pred_lines', 'confidence'/'pred_logits') in model output dict.")
    else:
        raise TypeError("Model forward returned an unexpected type. Adapt infer_outputs().")

    if pred_lines.dim() == 3 and pred_lines.size(0) == 1:
        pred_lines = pred_lines[0]
    if confidence.dim() == 2 and confidence.size(0) == 1:
        confidence = confidence[0]
    return {"pred_lines": pred_lines, "confidence": confidence}

def select_line(pred_lines: torch.Tensor, confidence: torch.Tensor, query_index: int, source: str = "query"):
    """
    Choose which line to draw:
      - 'query': draw the line at query_index
      - 'top1' : draw the most confident line
    Returns: (line_vec [3], score: float, idx: int)
    """
    if pred_lines.dim() == 3 and pred_lines.size(0) == 1:
        pred_lines = pred_lines[0]
    if confidence.dim() == 2 and confidence.size(0) == 1:
        confidence = confidence[0]

    K = pred_lines.size(0)
    if source == "top1":
        idx = int(torch.argmax(confidence).item())
    else:
        idx = max(0, min(query_index, K - 1))
    return pred_lines[idx], float(confidence[idx].item()), idx

def normalize01(a: np.ndarray):
    if a.size == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo + 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)

# ----- optional: try to fetch encoder token XY if you expose it on the model -----
def get_encoder_xy(model):
    """
    If your SymmetryTransformer stores encoder token coordinates (e.g., after PointNet++ SA),
    expose them as model.model.encoder_xy = [B,S,2] (or [S,2]) and this will pick them up.
    Return np.ndarray [S,2] or None if unavailable.
    """
    try:
        enc_xy = getattr(getattr(model, "model", model), "encoder_xy", None)
        if enc_xy is not None:
            enc_xy = enc_xy.detach().float().cpu().numpy()
            if enc_xy.ndim == 3:
                enc_xy = enc_xy[0]
            return enc_xy
    except Exception:
        pass
    return None

def fps_indices(pts_np: np.ndarray, S: int) -> np.ndarray:
    """Farthest Point Sampling indices from dense points to S proxies (for token viz when S != N)."""
    N = pts_np.shape[0]
    S = max(1, min(S, N))
    idxs = np.zeros(S, dtype=np.int64)
    centroid = pts_np.mean(axis=0, keepdims=True)
    d = np.linalg.norm(pts_np - centroid, axis=1)
    idxs[0] = int(np.argmax(d))
    dist = np.linalg.norm(pts_np - pts_np[idxs[0]], axis=1)
    for i in range(1, S):
        idxs[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(pts_np - pts_np[idxs[i]], axis=1))
    return idxs

def standardize_bucket_to_heads_tokens(A_layer: torch.Tensor, which_bucket: str, query_index: int) -> torch.Tensor:
    """
    Convert one layer's attention tensor to [H, S_like] (softmaxed) for visualization.
    Accepts the standardized shapes produced by attention_tools._to_BHTS:

      cross:        [B, H, Q, S]  -> pick row 'query_index' => [H, S]
      encoder_self: [B, H, S, S]  -> mean over queries (rows) => [H, S]
      decoder_self: [B, H, Q, Q]  -> pick row 'query_index' => [H, Q]
    """
    A0 = A_layer[0]  # [H,*,*]
    if A0.dim() != 3:
        raise ValueError(f"Unsupported attention tensor shape: {tuple(A_layer.shape)}")

    H, D1, D2 = A0.shape
    if which_bucket == "cross":
        q = max(0, min(query_index, D1 - 1))
        A_vis = A0[:, q, :]               # [H,S]
    elif which_bucket == "encoder_self":
        if D1 != D2:
            raise ValueError("encoder_self must be [H,S,S].")
        A_vis = torch.nanmean(A0, dim=1)  # [H,S]
    elif which_bucket == "decoder_self":
        if D1 != D2:
            raise ValueError("decoder_self must be [H,Q,Q].")
        q = max(0, min(query_index, D1 - 1))
        A_vis = A0[:, q, :]               # [H,Q]
    else:
        raise ValueError(f"Unknown bucket '{which_bucket}'")

    # make sure it's a proper distribution along tokens
    return torch.nn.functional.softmax(A_vis, dim=-1)

# ---------------------- Figure composer ----------------------

def draw_single_sample(
    out_path: str,
    pts_np: np.ndarray,                          # [N,2] original points
    line_vec_query: np.ndarray,                  # [3] (cos, sin, d) for the SAME query
    attn_layers: list,                           # list[(lname, torch[H,S_like])]
    which_bucket: str,
    query_index: int,
    enc_xy_np: np.ndarray = None                 # optional [S_like,2] true encoder-token coords
):
    """
    ONE figure per sample+query:
      - Top row: points + predicted line from the SAME query
      - Below: grid of attention maps [num_layers x num_heads], all layers, all heads
    """
    num_layers = len(attn_layers)
    num_heads = max((a.shape[0] for _, a in attn_layers), default=0)

    big_h = 1.3
    row_h = 3.0
    fig_h = 2.0 + big_h + row_h * max(1, num_layers)
    fig_w = 3.8 + 1.35 * max(1, num_heads)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)
    gs = GridSpec(nrows=(1 + max(1, num_layers)), ncols=max(1, num_heads),
                  height_ratios=[big_h] + [row_h] * max(1, num_layers),
                  figure=fig)

    # --- Main panel ---
    ax_main = fig.add_subplot(gs[0, :])
    print(pts_np.shape)
    ax_main.scatter(pts_np[:, 0], pts_np[:, 1], s=6, alpha=0.85, label="points")
    ax_main.set_aspect("equal")
    ax_main.set_xlim(-1, 1)
    ax_main.set_ylim(-1, 1)
    draw_lines_cos_sin_d(ax_main, line_vec_query[None, :], style="g--")
    ax_main.set_title(f"Predicted line from query {query_index}  |  bucket={which_bucket}", fontsize=10)
    ax_main.grid(True, alpha=0.2)

    # --- Attention grid ---
    if num_layers == 0 or num_heads == 0:
        fig.tight_layout(); ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, bbox_inches="tight"); plt.close(fig); return

    for li, (lname, A_vis) in enumerate(attn_layers, start=1):
        A_np = to_numpy(A_vis)  # [H, S_like]
        H_local, S_like = A_np.shape

        # choose coordinates per layer
        if which_bucket in ("cross", "encoder_self"):
            if enc_xy_np is not None and enc_xy_np.shape[0] == S_like:
                xy = enc_xy_np
            elif S_like == pts_np.shape[0]:
                xy = pts_np
            else:
                idxs = fps_indices(pts_np, S_like)  # proxy
                xy = pts_np[idxs]
        else:
            xy = None  # decoder_self -> not points

        for hi in range(num_heads):
            ax = fig.add_subplot(gs[li, hi]); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if hi < H_local:
                w = A_np[hi]  # [S_like], softmaxed
                if which_bucket in ("cross", "encoder_self"):
                    # sc = ax.scatter(xy[:, 0], xy[:, 1], c=w, s=16, edgecolor="none")
                    # fig.colorbar(sc, ax=ax, label="Attention weight")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    sc = ax.scatter(xy[:, 0], xy[:, 1], c=w, s=18, cmap="viridis",
                                    vmin=w.min(), vmax=max(1e-8, w.max()))
                    draw_lines_cos_sin_d(ax, line_vec_query[None, :], style="g-")
                    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
                else:
                    xs = np.linspace(-1, 1, w.shape[0]); ys = np.zeros_like(xs)
                    ax.scatter(xs, ys, c=w, s=18, cmap="viridis",
                               vmin=0.0, vmax=max(1e-8, w.max()))
                    ax.axhline(0, lw=0.5, alpha=0.3); ax.set_xlim(-1, 1); ax.set_ylim(-0.1, 0.1)

            if hi == 0:
                lbl = lname.split(".")[-3:]  # shorter label tail
                ax.set_ylabel("/".join(lbl), fontsize=7)
            if li == 1:
                ax.set_title(f"head {hi}", fontsize=8)

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

# ---------------------- Runner ----------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Save ONE PNG per sample+query with line & all heads from all layers.")
    p.add_argument("--ckpt", required=False, help="Path to .ckpt (e.g., lightning_logs/.../epoch=..ckpt)")
    p.add_argument("--out_dir", default="attn_viz_out", help="Directory to save figures")
    p.add_argument("--which_bucket", default="cross",
                   choices=["cross", "encoder_self", "decoder_self"],
                   help="Which attention to visualize")
    p.add_argument("--layer_substr", default=None,
                   help="Optional substring to filter layers (e.g., 'layers.1.multihead_attn')")
    p.add_argument("--num_samples", type=int, default=5, help="How many dataset samples to plot")
    p.add_argument("--query_index", type=int, default=0,
                   help="Which query to visualize (and which predicted line to draw).")
    p.add_argument("--line_source", choices=["query", "top1"], default="query",
                   help="Draw line from current 'query' (default) or the 'top1' most confident prediction.")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--use_train_loader", action="store_true",
                   help="Use train_dataloader() instead of test_dataloader() (for debugging parity with train.py)")
    p.add_argument("--dataset_path", default="pointNetpp_embeddings.npz",
                   help="Path to dataset file (for SymmetryDataset).")
    return p

def main():
    args = build_argparser().parse_args()

    # Repro
    pl.seed_everything(args.seed, workers=True)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # Build backbone exactly as in training
    backbone = get_model(13, input_features=9)

    # Load LightningModule from checkpoint
    model = SymmetryTransformer.load_from_checkpoint(
        args.ckpt,
        backbone=backbone,
        strict=True
    ).to(device)
    model.eval()

    # Acquire a dataloader from the module (match your current debugging flow if requested)
    # ds = SymmetryDataset(dataset_path)
    if args.use_train_loader:
        model.setup()
        dl = model.train_dataloader()
    else:
        try:
            dl = model.test_dataloader()
        except Exception:
            model.setup()
            dl = model.train_dataloader()

    if not dl:
        raise RuntimeError("No dataloader returned by the model.")
    dataset = SymmetryDataset(args.dataset_path)
    dataset.set_get_points_mode(True)  # <-- enable returning raw points if available
    n = len(dataset)
    if n == 0:
        raise RuntimeError("Empty dataset.")
    idxs = torch.randperm(n)[:args.num_samples].tolist()

    out_root = ensure_dir(args.out_dir)

    # try to fetch encoder-token XY coordinates once (if exposed)
    enc_xy_np_global = get_encoder_xy(model)

    for si, idx in enumerate(idxs):
        # Extract sample
        item = dataset[idx]
        print(item)
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            points, features = item[0], item[1]
        # elif isinstance(item, dict) and "points" in item:
        #     points = item["points"]
        # else:
        #     print(f"[WARN] Unexpected dataset item at idx {idx}: {type(item)}. Skipping.")
        #     continue

        # Shape [1, N, 2]
        if points.dim() == 2:
            points_b1 = points.unsqueeze(0).to(device)
            features_b1 = features.unsqueeze(0).to(device)
        elif points.dim() == 3 and points.size(0) == 1:
            points_b1 = points.to(device)
            features_b1 = features.to(device)
        else:
            points_b1 = points[:1].to(device)
            features_b1 = features[:1].to(device)

        # Forward pass with attention capture anywhere under the model (your utils)
        target_root = getattr(model, "model", model)
        with capture_any_transformer_attn(target_root) as store:
            if model.backbone_model == "pointnetpp":
                outputs = infer_outputs(model, features_b1)
            else:
                outputs = infer_outputs(model, points_b1)

        # Pull the chosen bucket
        buckets = store.get(args.which_bucket, {})
        if not buckets:
            print(f"[WARN] No attention captured for '{args.which_bucket}' on sample {idx}.")
            continue

        # Optionally filter by layer substring
        items = sorted(buckets.items(), key=lambda kv: kv[0])
        if args.layer_substr:
            items = [(k, v) for (k, v) in items if args.layer_substr in k]
            if not items:
                print(f"[WARN] After filtering, no layers match '{args.layer_substr}' for sample {idx}. Skipping.")
                continue

        # Choose line from SAME query (default) or top-1
        pred_lines = outputs["pred_lines"]   # [K,3]
        confid     = outputs["confidence"]   # [K]
        line_vec, line_score, chosen_idx = select_line(pred_lines, confid, args.query_index, source=args.line_source)
        for i, line_vec in enumerate(pred_lines):
            # Standardize each layer to [H, S_like] with SAME query index
            attn_layers = []
            for lname, A_bhts in items:
                try:
                    A_vis = standardize_bucket_to_heads_tokens(A_bhts, args.which_bucket, i)
                    attn_layers.append((lname, A_vis))
                except Exception as e:
                    print(f"[WARN] Layer '{lname}' unsupported ({tuple(A_bhts.shape)}): {e}")
                    continue
            if not attn_layers:
                print(f"[WARN] No layers could be visualized for sample {idx}.")
                continue

            pts_np = to_numpy(points_b1[0])
            out_path = os.path.join(out_root, f"{args.which_bucket}_q{args.query_index}_idx{str(idx).zfill(6)}_{i}.png")
            draw_single_sample(
                out_path=out_path,
                pts_np=pts_np,
                line_vec_query=to_numpy(line_vec),
                attn_layers=attn_layers,
                which_bucket=args.which_bucket,
                query_index=i,
                enc_xy_np=enc_xy_np_global
            )
            print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    main()
