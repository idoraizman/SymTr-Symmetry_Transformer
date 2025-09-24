import contextlib
from typing import Any, Dict, List, Tuple, Optional
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

# ----------------------------- Core capture utils -----------------------------

def _to_BHTS(attn: torch.Tensor, *, mha: nn.MultiheadAttention, query: torch.Tensor) -> torch.Tensor:
    """
    Deterministically convert attention weights to [B, H, T_q, T_k].
    Works across PyTorch versions that return:
      - [B, H, T, S] (new)
      - [B*H, T, S] (legacy when average_attn_weights=False)
      - [T, S]      (averaged over batch & heads)
    """
    H = mha.num_heads
    batch_first = getattr(mha, "batch_first", False)

    if batch_first:
        # query: [B, T_q, E]
        B, Tq = query.shape[:2]
    else:
        # query: [T_q, B, E]
        Tq, B = query.shape[:2]

    if attn.dim() == 4:
        # assume already [B, H, Tq, Tk]
        return attn

    if attn.dim() == 3:
        # [B*H, Tq, Tk]
        BH, Tq2, Tk = attn.shape
        if Tq2 != Tq:
            raise RuntimeError(f"Shape mismatch: Tq (query)={Tq} vs attn.Tq={Tq2}")
        if BH % H != 0:
            raise RuntimeError(f"Cannot reshape BH={BH} into H={H}")
        B2 = BH // H
        if B2 != B:
            raise RuntimeError(f"Inferred B={B2} != actual B={B}")
        return attn.view(B, H, Tq, Tk)

    if attn.dim() == 2:
        # [Tq, Tk] averaged across batch & heads
        Tq2, Tk = attn.shape
        if Tq2 != Tq:
            raise RuntimeError(f"Shape mismatch: Tq (query)={Tq} vs attn.Tq={Tq2}")
        return attn.unsqueeze(0).unsqueeze(0)  # [1, 1, Tq, Tk]

    # Fallback (shouldn't happen)
    return attn


def _build_parent_map(root: nn.Module):
    """
    Map: child_module -> (qualified_parent_name, parent_module, child_name_in_parent)
    Useful to infer bucket and stable names.
    """
    parent_of: Dict[nn.Module, Tuple[str, nn.Module, str]] = {}
    for pname, parent in root.named_modules():
        for cname, child in parent.named_children():
            parent_of[child] = (pname, parent, cname)
    return parent_of


def _bucket_for(mha: nn.MultiheadAttention, parent_of) -> str:
    """
    Decide which bucket an MHA belongs to by looking at its parent layer & child name.
    """
    pname, parent, cname = parent_of.get(mha, ("", None, ""))
    if isinstance(parent, nn.TransformerEncoderLayer):
        return "encoder_self"
    if isinstance(parent, nn.TransformerDecoderLayer):
        if cname == "self_attn":
            return "decoder_self"
        if cname == "multihead_attn":
            return "cross"
        return "decoder_other"
    return "unknown"


@contextlib.contextmanager
def capture_any_transformer_attn(root: nn.Module):
    """
    Context manager: patch ALL nn.MultiheadAttention under `root`,
    force per-head attention (need_weights=True, average_attn_weights=False),
    and store them per bucket & qualified layer name.

    Yields:
        store: Dict[str, Dict[str, torch.Tensor]]
            bucket -> { qualified_layer_name : attn[B,H,T,S] (on device) }
    """
    store: Dict[str, Dict[str, torch.Tensor]] = {}
    originals: List[Tuple[nn.Module, Any]] = []
    parent_of = _build_parent_map(root)

    def patch(name: str, mha: nn.MultiheadAttention):
        orig = mha.forward
        bucket = _bucket_for(mha, parent_of)
        store.setdefault(bucket, {})

        def wrapped(query, key, value, *args, **kwargs):
            kwargs["need_weights"] = True
            # best-effort keep per-head weights (older PT may not support the kw)
            try:
                kwargs.setdefault("average_attn_weights", False)
            except Exception:
                pass

            out, attn = orig(query, key, value, *args, **kwargs)

            try:
                attn_bhts = _to_BHTS(attn, mha=mha, query=query).detach()
                store[bucket][name] = attn_bhts
            except Exception:
                # Don't break forward if capturing fails
                pass

            return out, attn

        mha.forward = wrapped  # type: ignore[method-assign]
        originals.append((mha, orig))

    # Patch every MHA under root
    for name, m in root.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            patch(name, m)

    try:
        yield store
    finally:
        # Restore originals and move captured maps to CPU
        for m, orig in originals:
            m.forward = orig  # type: ignore[method-assign]
        for b in store:
            for k in list(store[b].keys()):
                store[b][k] = store[b][k].cpu()


@torch.no_grad()
def extract_attention_maps_anywhere(
    model: nn.Module,
    inputs: Any,
    *,
    return_outputs: bool = False,
    target_submodule: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Run a forward pass on `model(inputs)` while capturing attention weights from ANY
    nn.MultiheadAttention under `target_submodule` (defaults to `model`).

    Returns:
        {
          'buckets': Dict[str, Dict[str, Tensor[B,H,T,S]]],
          'outputs': optional model outputs
        }
    """
    was_training = model.training
    model.eval()

    root = target_submodule if target_submodule is not None else model

    with capture_any_transformer_attn(root) as store:
        outputs = model(inputs)

    if was_training:
        model.train()

    result: Dict[str, Any] = {"buckets": store}
    if return_outputs:
        result["outputs"] = outputs
    return result

# ----------------------------- Visualization utils ----------------------------

def visualize_attention_on_points(
    points: torch.Tensor,
    attn_bhts: torch.Tensor,
    *,
    query_index: int = 0,
    head: Optional[int] = None,
    title: Optional[str] = None,
    figsize=(5, 5),
):
    """Create a scatter plot where point color encodes attention weights for a given query."""
    matplotlib.use("Agg")  # headless-safe

    # Debug shapes
    print(f"Points shape: {points.shape}")
    print(f"Attention tensor shape: {attn_bhts.shape}")

    # points: [B,N,2] -> [N,2]
    if points.dim() == 3:
        points = points[0]  # Take first batch
    pts = points.detach().cpu()

    # attn_bhts: [B,H,T,S] -> [H,T,S]
    if attn_bhts.dim() != 4:
        raise ValueError(f"Expected 4D attention tensor, got shape {attn_bhts.shape}")
    
    A = attn_bhts[0]  # [H,T,S]
    print(f"A shape after batch selection: {A.shape}")

    # Average across heads or take specific head -> [T,S]
    if head is None:
        weights = A.mean(dim=0)  # [T,S]
    else:
        weights = A[head]  # [T,S]
    
    print(f"Weights shape before query selection: {weights.shape}")
    
    # Handle the case where weights are [1,1]
    if weights.shape == (1, 1):
        # Expand weights to match number of points
        weights = weights.expand(1, pts.shape[0])
    
    # Ensure weights match points
    if weights.shape[1] != pts.shape[0]:
        raise ValueError(
            f"Attention weights size {weights.shape[1]} doesn't match number of points {pts.shape[0]}.\n"
            f"Weights shape: {weights.shape}, Points shape: {pts.shape}"
        )

    # choose row = query token
    query_index = max(0, min(query_index, weights.shape[0] - 1))
    w = weights[query_index]  # [S]
    print("sum over keys:", float(w.sum()))   # ~1.0
    print("min/max:", float(w.min()), float(w.max()))

    # normalize for colormap
    w = (w - w.min()) / (w.max() - w.min() + 1e-6)
    x = pts[:, 0].numpy()
    y = pts[:, 1].numpy()
    c = w.numpy()

    # Create visualization
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=c, s=16, edgecolor="none")
    fig.colorbar(sc, ax=ax, label="Attention weight")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig