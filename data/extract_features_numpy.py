import argparse
import numpy as np
import torch
from tqdm import tqdm
from model.pointNetpp_pretrained import get_model

N = 2048     # number of points per sample (pad/truncate to this)
MAX_L = 2  # max number of lines per sample (pad/truncate to this)

@torch.inference_mode()
def extract_pointnetpp_to_npz(
    input_file: str,
    output_path: str,
    checkpoint_path: str = "model/checkpoint_pointnetpp.pth",
    batch_size: int = 8,
    fp16: bool = True,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load input npz
    data = np.load(input_file, allow_pickle=True)
    points_list = list(data["points"])   # list of [Ni, D]
    lines_list  = list(data["lines"])    # list of [Li, K]
    B = len(points_list)

    # ---- infer input D
    D = points_list[0].shape[1] if len(points_list) else 3

    # ---- model
    model = get_model(13, input_features=9).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- probe to find feature dimension C
    arr = np.asarray(points_list[0], dtype=np.float32)
    x = torch.from_numpy(arr[None, ...]).to(device)
    if D == 2:
        z = torch.zeros((1, x.shape[1], 1), device=device)
        xyz = torch.cat([x, z], dim=-1)
    else:
        xyz = x
    rgb = torch.zeros((1, x.shape[1], 3), device=device)
    inp = torch.cat([xyz, rgb, xyz], dim=-1).permute(0, 2, 1).contiguous()
    out, _ = model(inp)
    if out.shape[1] < out.shape[2]:
        out = out.permute(0, 2, 1)
    C = int(out.shape[-1])

    # ---- allocate dense arrays
    features_arr = np.zeros((B, N, C), dtype=np.float16 if fp16 else np.float32)
    points_arr   = np.zeros((B, N, D), dtype=np.float32)
    lines_arr    = np.zeros((B, MAX_L, 4), dtype=np.float32)   # assuming lines = [K=4] coords?
    lens_points  = np.zeros(B, dtype=np.int32)
    lens_lines   = np.zeros(B, dtype=np.int32)

    # ---- process in batches
    num_batches = (B + batch_size - 1) // batch_size
    for s in tqdm(range(0, B, batch_size), total=num_batches, desc="Extracting features", leave=False):
        e = min(s + batch_size, B)
        batch_size_cur = e - s

        # build padded point batch [b, N, D]
        x_np = np.zeros((batch_size_cur, N, D), dtype=np.float32)
        for i in range(batch_size_cur):
            arr = np.asarray(points_list[s + i], dtype=np.float32)
            n = min(arr.shape[0], N)
            lens_points[s + i] = n
            x_np[i, :n, :] = arr[:n]

        x = torch.from_numpy(x_np).to(device)

        # model input
        if D == 2:
            z = torch.zeros((x.shape[0], x.shape[1], 1), device=device)
            xyz = torch.cat([x, z], dim=-1)
        else:
            xyz = x
        rgb = torch.zeros((x.shape[0], x.shape[1], 3), device=device)
        inp = torch.cat([xyz, rgb, xyz], dim=-1).permute(0, 2, 1).contiguous()

        # run model
        out, _ = model(inp)
        if out.shape[1] < out.shape[2]:
            out = out.permute(0, 2, 1)
        out = out.detach().cpu().numpy()  # [b, N, C]

        # save features and points
        features_arr[s:e, :, :] = out.astype(features_arr.dtype)
        points_arr[s:e, :, :]   = x_np

        # process lines
        for i in range(batch_size_cur):
            li = np.asarray(lines_list[s + i], dtype=np.float32)
            if li.ndim == 1:
                li = li[None, :]   # at least 2D
            L = min(li.shape[0], MAX_L)
            lens_lines[s + i] = L
            lines_arr[s + i, :L, :li.shape[1]] = li[:L]

        # cleanup
        del x, inp, out

    # ---- save all arrays to NPZ
    np.savez_compressed(
        output_path,
        features=features_arr,
        points=points_arr,
        lines=lines_arr,
        lens_points=lens_points,
        lens_lines=lens_lines,
    )

    print(f"[ok] wrote {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset.npz")
    ap.add_argument("--output", default="pointNetpp_embeddings.npz")
    ap.add_argument("--checkpoint", default="model/checkpoint_pointnetpp.pth")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    extract_pointnetpp_to_npz(args.input, args.output, args.checkpoint, args.batch_size, args.fp16, args.device)
