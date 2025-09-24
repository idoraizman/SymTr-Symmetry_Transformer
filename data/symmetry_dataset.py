import numpy as np
import torch
from torch.utils.data import Dataset

class SymmetryDataset(Dataset):
    """
    Dataset for symmetry tasks from .npz file (preprocessed with fixed N, MAX_L).
    
    Expects npz keys:
      - 'features'    : float [B, N, C]
      - 'points'      : float [B, N, D]
      - 'lines'       : float [B, MAX_L, K]
      - 'lens_points' : int   [B]
      - 'lens_lines'  : int   [B]

    __getitem__(i) -> (features, lines)
      or, if get_points_mode=True: (points, features, lines)
    """

    def __init__(self, path: str, ret_features: bool = True, get_points_mode: bool = False):
        if not path.endswith(".npz"):
            raise ValueError("This version only supports .npz files")

        self.path = path
        # use memory mapping to avoid loading whole file eagerly
        self.data = np.load(path, mmap_mode="r")

        self.features    = self.data["features"]
        self.points      = self.data["points"]
        self.lines       = self.data["lines"]
        self.lens_points = self.data["lens_points"]
        self.lens_lines  = self.data["lens_lines"]

        self.n = self.features.shape[0]
        self.get_points_mode = get_points_mode
        self.ret_features = ret_features

    def set_get_points_mode(self, mode: bool):
        self.get_points_mode = mode

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # slice directly (no Python list overhead)
        feats = torch.from_numpy(self.features[i])
        lines = torch.from_numpy(self.lines[i][: self.lens_lines[i]])
        pts = torch.from_numpy(self.points[i][: self.lens_points[i]])
        if self.get_points_mode:
            return pts, feats, lines
        if self.ret_features:
            return feats, lines
        return pts, lines
