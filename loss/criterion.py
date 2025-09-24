import torch
from torch import nn
import torch.nn.functional as F
from loss.matcher import HungarianMatcher
from configs.config import DIRECTION_LOSS_WEIGHT, CONF_LOSS_WEIGHT

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, losses):
        """ Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.losses = losses
        self.matcher = HungarianMatcher()
    
    def loss_lines(self, pred_lines, confidence, targets, indices, log=True,
              lambda_dir: float = DIRECTION_LOSS_WEIGHT,    # weight for direction term
              lambda_off: float = 1.0,    # weight for offset term
              lambda_unit: float = 0.1,   # unit-norm regularizer on (cos,sin)
              eps: float = 1e-12,
              **kwargs):
        """
        Lines encoded as (cos, sin, d[, ...]) with equation n·x + d = 0.
        """
        # total GT lines across batch
        M = sum(t.shape[0] for t in targets)
        if M == 0:
            z = pred_lines.sum() * 0.0
            return {'loss_lines': z}

        # gather matched pairs
        idx = self._get_src_permutation_idx(indices)          # tuple for [B,P,*] indexing
        tgt = torch.cat([t[J][..., :3] for t, (_, J) in zip(targets, indices)], dim=0)  # [M,3]
        src = pred_lines[idx][..., :3]                        # [M,3] -> (coŝ, sin̂, d̂)

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

        # offset (sign-invariant to (n,d) ↔ (-n,-d))
        # off_term = torch.minimum((pd - td).abs(), (pd + td).abs())
        off_term = (pd - td).abs()
        loss_off = off_term.mean()

        # unit-norm regularizer on predicted (cos,sin)
        # loss_unit = F.mse_loss(pn_norm.squeeze(1), torch.ones_like(pn_norm.squeeze(1)))

        # total = lambda_dir * loss_dir + lambda_off * loss_off + lambda_unit * loss_unit
        total = (lambda_dir * loss_dir + lambda_off * loss_off) / (lambda_dir + lambda_off)
        return {'loss_lines': total}
    
    def loss_confidence(self, pred_lines, confidence, targets, indices, **kwargs):
        """
        BCE-with-logits over confidence:
        - matched predictions -> target 1
        - unmatched predictions -> target 0
        confidence: [B, K] logits
        """

        # Targets: same shape as confidence
        target_conf = torch.zeros_like(confidence)  # [B, K]

        # Mark matched predictions as 1
        for b, (src_idx, _) in enumerate(indices):
            src_idx = torch.as_tensor(src_idx, device=confidence.device, dtype=torch.long)
            if src_idx.numel():
                target_conf[b, src_idx] = 1.0

        # Mean BCE over all logits (stable scale; handles no-GT case)
        loss_conf = F.binary_cross_entropy_with_logits(confidence, target_conf, reduction='mean')
        return {"loss_conf": loss_conf}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, pred_lines, confidence, targets, indices, **kwargs):
        loss_map = {
            'lines': self.loss_lines,
            'confidence': self.loss_confidence,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](pred_lines, confidence, targets, indices, **kwargs)


    def forward(self, pred_lines, confidence, targets, direction_loss_weight=DIRECTION_LOSS_WEIGHT, conf_loss_weight=CONF_LOSS_WEIGHT):
        """ This performs the loss computation.
        Parameters:
            pred_lines: Tensor of shape [B, num_preds, 4] — predicted lines
            confidence: Tensor of shape [B, num_preds] — predicted confidence scores
        """
        indices = self.matcher(pred_lines, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, pred_lines, confidence, targets, indices, lambda_dir=direction_loss_weight))

        loss = (losses["loss_lines"] + conf_loss_weight * losses["loss_conf"]) / (1+conf_loss_weight)
        return loss