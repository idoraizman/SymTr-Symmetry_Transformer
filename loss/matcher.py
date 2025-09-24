import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching for 2D lines represented as (nx, ny, d) with line: n·x + d = 0.
    - Sign-invariant: (n, d) and (-n, -d) are identical lines.
    - Cost = w_dir * (1 - |n̂ · n|) + w_d * min(|d̂ - d|, |d̂ + d|)
    """

    def __init__(self, w_dir: float = 1.0, w_d: float = 1.0, canonicalize: bool = True, eps: float = 1e-12):
        super().__init__()
        self.w_dir = w_dir
        self.w_d = w_d
        self.canonicalize = canonicalize
        self.eps = eps

    @torch.no_grad()
    def forward(self, preds: torch.Tensor, targets: list[torch.Tensor]):
        """
        Args:
            preds:   [B, P, 3] predictions (nx, ny, d) per batch item
            targets: list of length B; each element is a tensor [K_i, 3] of GT lines

        Returns:
            indices: list of length B; each is a tuple (idx_pred, idx_tgt) with matched indices
        """
        B, P, D = preds.shape
        assert D == 3, f"Expected last dim=3 (nx, ny, d), got {D}"

        indices = []

        for b in range(B):
            pred_b = preds[b].detach()     # [P, 3]
            tgt_b  = targets[b].detach()   # [K, 3]  (can be K=0)
            if tgt_b.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
                continue

            # split + normalize normals to unit length
            pn, pd = pred_b[:, :2], pred_b[:, 2]
            tn, td = tgt_b[:, :2],  tgt_b[:, 2]

            pn = pn / (pn.norm(dim=1, keepdim=True) + self.eps)   # [P,2]
            tn = tn / (tn.norm(dim=1, keepdim=True) + self.eps)   # [K,2]
            pd = pd / (pn.norm(dim=1, keepdim=True).squeeze(1) + self.eps)  # scale d by predicted normal length
            td = td / (tn.norm(dim=1, keepdim=True).squeeze(1) + self.eps)  # scale d by GT normal length
            # optional canonicalization: flip (n,d) so d >= 0 for both sets
            # if self.canonicalize:
            #     maskp = pd < 0
            #     pn[maskp] = -pn[maskp]
            #     pd = torch.where(maskp, -pd, pd)

            #     maskt = td < 0
            #     tn[maskt] = -tn[maskt]
            #     td = torch.where(maskt, -td, td)

            # ---- build cost matrix [P, K] ----
            # direction cost: 1 - |pn · tn|
            dot = pn @ tn.t()                           # [P,K]
            dir_cost = (1.0 - dot).abs() # [P,K]

            # distance cost: min(|pd - td|, |pd + td|)
            # broadcast to [P,K]
            pd_col = pd[:, None]
            td_row = td[None, :]
            d_cost = (pd_col - td_row).abs()      # [P,K]
            cost = self.w_dir * dir_cost + self.w_d * d_cost      # [P,K]

            # solve LSAP on CPU
            cost_np = cost.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))

        return indices

