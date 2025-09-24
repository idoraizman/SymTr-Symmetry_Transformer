import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from model.SymTr import SymTr
from loss.criterion import SetCriterion
from data.symmetry_dataset import SymmetryDataset
from configs.config import BATCH_SIZE, POINTNET_OUTPUT_DIM, SCHEDULER_GAMMA, SCHEDULER_STEP_SIZE, WEIGHT_DECAY, DIRECTION_LOSS_WEIGHT, HIDDEN_DIM, NUM_WORKERS, RANDOM_STATE, CONF_LOSS_WEIGHT, DATASET_PATH, NUM_HEADS
import os, random, numpy as np

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA determinism
random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
pl.seed_everything(RANDOM_STATE, workers=True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

class SymmetryTransformer(pl.LightningModule):
    def __init__(self, hidden_dim=HIDDEN_DIM, nheads=NUM_HEADS, num_encoder_layers=6, num_decoder_layers=6, input_dim=2, num_preds=4,
                 learning_rate=1e-3, output_dim=4, direction_loss_weight=DIRECTION_LOSS_WEIGHT, conf_loss_weight=CONF_LOSS_WEIGHT, 
                 dataset_path=DATASET_PATH, backbone_model="pointnetpp", no_attn=False):
        super().__init__()
        self.save_hyperparameters()
        # Initialize the DETR model
        self.model = SymTr(
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            input_dim=input_dim,
            num_preds=num_preds,
            output_dim=output_dim,
            backbone_model=backbone_model,
            no_attn=no_attn
        )
        
        self.learning_rate = learning_rate
        self.direction_loss_weight = direction_loss_weight
        self.criterion = SetCriterion(['lines', 'confidence'])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.conf_loss_weight = conf_loss_weight
        self.dataset_path = dataset_path
        self.backbone_model = backbone_model

    def setup(self, stage=None):
        print(f"Loading dataset from {self.dataset_path}")
        dataset = SymmetryDataset(self.dataset_path, ret_features=(self.backbone_model=="pointnetpp"))
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        points, targets = batch
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
        pred_lines, confidence = self(points)
        loss = self.criterion(pred_lines=pred_lines, confidence=confidence, targets=targets, direction_loss_weight=self.direction_loss_weight, conf_loss_weight=self.conf_loss_weight)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss
    
    def validation_step(self, batch):
        points, targets = batch
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
        pred_lines, confidence = self(points)
        loss = self.criterion(pred_lines, confidence, targets, direction_loss_weight=self.direction_loss_weight)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss
    
    def test_step(self, batch):
        points, targets = batch
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
        pred_lines, confidence = self(points)
        loss = self.criterion(pred_lines, confidence, targets, direction_loss_weight=self.direction_loss_weight)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    @staticmethod
    def custom_collate_fn(batch):
        xs, ys = zip(*batch)  # unzip (x, y) pairs
        xs = torch.stack(xs)  # stack x into tensor of shape (B, ...)
        return xs, list(ys)   # keep y as list of length B
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=self.custom_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=self.custom_collate_fn)
