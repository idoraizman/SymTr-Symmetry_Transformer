import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.lightning_model import SymmetryTransformer
from loss.criterion import HungarianMatcher

"""
To run evaluation:
    python3 calculate_metrics.py --ckpt path/to/checkpoint.ckpt
Outputs average direction and offset losses on the test set.
"""

def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def _calculate_losses(pred_lines, targets, indices,eps: float = 1e-12):
        """
        Lines encoded as (cos, sin, d[, ...]) with equation n·x + d = 0.
        """
        # total GT lines across batch
        M = sum(t.shape[0] for t in targets)
        if M == 0:
            z = pred_lines.sum() * 0.0
            return {'loss_lines': z}

        # gather matched pairs
        idx = _get_src_permutation_idx(indices)  # tuple for [B,P,*] indexing
        tgt = torch.cat([t[J][..., :3] for t, (_, J) in zip(targets, indices)], dim=0)  # [M,3]
        src = pred_lines[idx][..., :3]      # [M,3] -> (coŝ, sin̂, d̂)

        # split + normalize normals
        pn, pd = src[:, :2], src[:, 2]
        tn, td = tgt[:, :2], tgt[:, 2]

        pn_norm = pn.norm(dim=1, keepdim=True).clamp_min(eps)
        tn_norm = tn.norm(dim=1, keepdim=True).clamp_min(eps)
        pn_u = pn / pn_norm
        tn_u = tn / tn_norm  # GT should already be unit; normalize for safety
        pd = pd / pn_norm.squeeze(1)  # scale d by predicted normal length
        td = td / tn_norm.squeeze(1)  # scale d by GT normal length

        # direction (sign-invariant): 1 - |cos θ|
        dir_term = (1.0 - (pn_u * tn_u).sum(dim=1)).abs()
        loss_dir = dir_term.mean()

        off_term = (pd - td).abs()
        loss_off = off_term.mean()

        return loss_dir, loss_off

def calculate_metrics(model, dataloader: DataLoader, max_samples=None):
    model.eval()
    matcher = HungarianMatcher()
    total_dir_loss = 0.0
    total_off_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if max_samples is not None and i >= max_samples:
                break
            points, targets = batch
            points = points.to(model.device)
            targets = [t.to(model.device) for t in targets]
            valid_points = []
            valid_targets = []
            for p, t in zip(points, targets):
                try:
                    tn, td = t[:, :2],  t[:, 2]
                    valid_points.append(p)
                    valid_targets.append(t)
                except:
                    print("Error in targets, skipping sample")

            if not valid_points:
                return None
            points = torch.stack(valid_points)
            targets = valid_targets
            pred_lines, _ = model(points)
            
            indices = matcher(pred_lines, targets)
            
            dir_loss, off_loss = _calculate_losses(pred_lines, targets, indices)
            
            batch_size = len(valid_points)
            total_dir_loss += dir_loss.item() * batch_size
            total_off_loss += off_loss.item() * batch_size
            total_samples += batch_size

    avg_dir_loss = total_dir_loss / total_samples if total_samples > 0 else 0.0
    avg_off_loss = total_off_loss / total_samples if total_samples > 0 else 0.0

    metrics = {
        'avg_direction_loss': avg_dir_loss,
        'avg_offset_loss': avg_off_loss,
    }
    return metrics

def main():
    # load model, dataloader, device setup here
    model = SymmetryTransformer.load_from_checkpoint(
        args.ckpt,
        strict=True  # set to False if you changed any arg names/shapes
    )
    model.setup(stage="test")
    dl = model.test_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = calculate_metrics(model, dl, max_samples=100)
    print(metrics)

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint")
    args = ap.parse_args()
    main()