import os
import sys
import importlib
from typing import Optional

import torch
import torch.nn as nn

#models_root = "/home/temp_compute/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch/models"
#ckpt_path = "/home/temp_compute/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch/log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth"

class YanxPointNet2Encoder(nn.Module):
    """
    Wrapper around yanx27/Pointnet_Pointnet2_pytorch classification model
    (pointnet2_cls_ssg) to use it as a frozen feature extractor.

    It:
      - loads the pretrained checkpoint (best_model.pth)
      - hooks the fc2 layer (256-D) as a global feature
      - returns [B, out_dim] features for [B, N, 3] or [B, N, 6] inputs
    """

    def __init__(
        self,
        ckpt_path: str = "/home/bweiss/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth",
        models_root: Optional[str] = "/home/bweiss/Benedikt/ShapeDream/models/Pointnet_Pointnet2_pytorch",
        model_module: str = "models.pointnet2_cls_ssg",
        num_classes: int = 40,
        normal_channel: bool = False,
        out_dim: int = 256,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            ckpt_path: path to yanx best_model.pth
                       (e.g. /path/to/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_cls_ssg/checkpoints/best_model.pth)
            models_root: folder that contains the yanx `models/` directory.
                         Example: /path/to/Pointnet_Pointnet2_pytorch/models
                         If None, assumes module is already importable.
            model_module: module name inside `models_root` (default: "pointnet2_cls_ssg")
            num_classes: number of classes the classifier was trained on (ModelNet40 → 40)
            normal_channel: True if the model was trained with normals (xyz+normal → 6 channels)
            out_dim: output feature dimension (defaults to 256, the fc2 output size)
            device: "cuda" or "cpu"
        """
        super().__init__()

        # Make sure we can import yanx models
        if models_root is not None:
            sys.path.insert(0, models_root)

        MODEL = importlib.import_module(model_module)

        # Instantiate classifier
        self.backbone = MODEL.get_model(
            num_classes, normal_channel=normal_channel
        )

        # Load pretrained checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # yanx checkpoints store model weights under 'model_state_dict'
        self.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)

        self.backbone.to(device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.device = torch.device(device)
        self.normal_channel = normal_channel

        # We'll hook the fc2 layer to capture a 256-D feature vector
        self._feat = None

        def _hook_fc2(module, inp, out):
            # out is [B, 256]
            self._feat = out

        if not hasattr(self.backbone, "fc2"):
            raise AttributeError(
                "Expected backbone to have an 'fc2' attribute. "
                "Check that you're using pointnet2_cls_ssg from yanx repo."
            )

        self.backbone.fc2.register_forward_hook(_hook_fc2)

        # Optionally project to a different dimension
        self.in_feat_dim = 256  # fc2 output dimension in yanx pointnet2_cls_ssg
        if out_dim is None or out_dim == self.in_feat_dim:
            self.proj = nn.Identity()
            self.out_dim = self.in_feat_dim
        else:
            self.proj = nn.Linear(self.in_feat_dim, out_dim)
            self.out_dim = out_dim

    @torch.no_grad()
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, 3] or [B, N, 6] (xyz or xyz+normal)

        Returns:
            features: [B, out_dim]
        """
        # Expect [B, N, C]. The yanx model expects [B, C, N].
        if points.ndim != 3:
            raise ValueError(f"Expected points of shape [B, N, C], got {points.shape}")

        B, N, C = points.shape
        if self.normal_channel:
            if C != 6:
                raise ValueError(
                    f"normal_channel=True but input has {C} channels (expected 6: xyz+normal)"
                )
        else:
            if C != 3:
                raise ValueError(
                    f"normal_channel=False but input has {C} channels (expected 3: xyz)"
                )

        # [B, N, C] -> [B, C, N]
        x = points.to(self.device).permute(0, 2, 1).contiguous()

        self._feat = None
        out = self.backbone(x)

        # Some variants may return (logits, extras)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out  # [B, num_classes], not used here

        if self._feat is None:
            raise RuntimeError(
                "Forward hook on fc2 did not run. "
                "Check that the backbone forward uses fc2."
            )

        feat = self._feat  # [B, 256]
        feat = self.proj(feat)  # [B, out_dim]
        return feat
