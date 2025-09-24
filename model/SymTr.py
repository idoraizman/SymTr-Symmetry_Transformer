import torch
from torch import nn
from model.pointnet import PointNet2DWithTransform
from configs.config import POINTNET_OUTPUT_DIM, POINTNETPP_OUTPUT_DIM

class SymTr(nn.Module):
    """
    Symmetry Transformer model for detecting symmetry lines in 2D point clouds.
    Combines a backbone network (PointNet, PointNet++, or none) with a Transformer architecture.
    Inputs:
        - hidden_dim: Dimension of the transformer hidden layers.
        - nheads: Number of attention heads in the transformer.
        - num_encoder_layers: Number of encoder layers in the transformer.
        - num_decoder_layers: Number of decoder layers in the transformer.
        - input_dim: Dimension of input points (2 for 2D points).
        - num_preds: Number of symmetry line predictions to make.
        - output_dim: Dimension of each prediction (e.g., 3 for (nx, ny, d) + 1 for confidence).
        - backbone_model: Which backbone to use ("pointnet", "pointnetpp", or "None").
    """
    
    def __init__(self, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, 
                 input_dim=2, num_preds=4, output_dim=4, backbone_model="pointnetpp", no_attn=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_preds = num_preds
        self.output_dim = output_dim
        self.backbone_model = backbone_model
        self.no_attn = no_attn
        
        self.input_queries = nn.Parameter(torch.randn(num_preds, hidden_dim))

        self.backbone = None
        if backbone_model == "pointnetpp":
            self.input_proj = nn.Linear(POINTNETPP_OUTPUT_DIM, hidden_dim)
        elif backbone_model == "pointnet":
            self.backbone = PointNet2DWithTransform(input_dim=input_dim, output_dim=POINTNET_OUTPUT_DIM)
            self.input_proj = nn.Linear(POINTNET_OUTPUT_DIM+input_dim, hidden_dim)
        elif backbone_model == "None":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # create a default PyTorch transformer
        if no_attn:
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                PointNet2DWithTransform(input_dim=hidden_dim, output_dim=hidden_dim),
            )
            output_dim = output_dim * num_preds  # adjust output dim for backbone only mode
        else:
            self.transformer = nn.Transformer(
                hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=False)

        # prediction heads
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            ) # values are (nx, ny, c, confidence_score)

    def forward(self, inputs):
        B = inputs.shape[0]
        
        if self.backbone_model == "pointnet":
            backbone_features = self.backbone(inputs)  # [B, N, POINTNET_OUTPUT_DIM]
            inputs = torch.cat([inputs, backbone_features.unsqueeze(1).expand(-1, inputs.shape[1], -1)], dim=-1)
            
        x = self.input_proj(inputs)                   # [B, N, hidden_dim]
        # Transformer wants [S, B, E]
        if self.no_attn:
            x = self.pred_head(x)                        # [B, hidden_dim]
            outputs = self.output_proj(x).view(B, self.num_preds, self.output_dim)  # [B, num_preds, output_dim]
        else:
            src = (x).permute(1, 0, 2)                   # [N, B, hidden] TODO: maybe 0.1 * x

            # Queries (learned)
            tgt = self.input_queries.unsqueeze(1).expand(-1, B, -1)  # [num_queries, B, hidden]

            # the feature number of src and tgt must be equal to d_model
            assert(src.shape[2] == self.hidden_dim and tgt.shape[2] == self.hidden_dim)
            
            h = self.transformer(src, tgt).transpose(0, 1)   # [B, num_queries, hidden]
            outputs = self.output_proj(h)
        
        confidence = outputs[..., -1]                # [...,]
        pred = outputs[..., :-1]                     # [..., 3] = (a,b,c)

        a = pred[..., 0]
        b = pred[..., 1]
        c = pred[..., 2]

        # normalize the normal and scale the constant accordingly
        norm = torch.sqrt(a * a + b * b + 1e-6)      # [...,]
        n1 = a / norm                               
        n2 = b / norm
        d  = c / norm

        # enforce n2 > 0 by flipping sign where n2 < 0
        flip_mask = (n2 < 0)                         # bool tensor [...]
        # flip n1,n2,d where mask is True
        n1 = torch.where(flip_mask, -n1, n1)
        n2 = torch.where(flip_mask, -n2, n2)
        d  = torch.where(flip_mask, -d, d)

        pred_lines = torch.stack([n1, n2, d], dim=-1)  # [..., 3]
        return pred_lines, confidence
    