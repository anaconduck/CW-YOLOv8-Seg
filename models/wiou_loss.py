"""
Wise-IoU (WIoU): Bounding Box Regression Loss with Dynamic Focusing Mechanism
Supports WIoU v1, v2, and v3.
Reference: Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism", 2023.
Integrated for Ultralytics YOLOv8 segmentation training.
"""

import math
import torch
import torch.nn as nn


class WiseIoULoss(nn.Module):
    """
    Wise-IoU Loss with Dynamic Non-Monotonic Focusing Mechanism (WIoU v1, v2, v3).

    Args:
        version (int): 1, 2, or 3 (default: 3).
        alpha (float): Hyperparameter alpha for WIoU v3 non-monotonic mapping (default: 1.9).
        delta (float): Hyperparameter delta for WIoU v3 outlier threshold (default: 3.0).
        momentum (float): Moving average momentum for running mean of IoU loss (default: 0.9).
        eps (float): Epsilon to prevent division by zero.
    """
    def __init__(
        self,
        version: int = 3,
        alpha: float = 1.9,
        delta: float = 3.0,
        momentum: float = 0.9,
        eps: float = 1e-7
    ):
        super().__init__()
        assert version in [1, 2, 3], f"WIoU version must be 1, 2, or 3. Got {version}."
        self.version = version
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self.eps = eps

        # Running mean of IoU loss for normalization in v2 and v3
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculates WIoU loss.
        Args:
            pred_boxes (torch.Tensor): Predicted boxes in (x1, y1, x2, y2) format, shape (N, 4).
            target_boxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (N, 4).
        Returns:
            torch.Tensor: WIoU loss values (N,).
        """
        # Box coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.chunk(4, -1)

        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union area
        b1_area = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        b2_area = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        union_area = b1_area + b2_area - inter_area + self.eps

        # Standard IoU & IoU Loss
        iou = (inter_area / union_area).clamp(min=0, max=1.0)
        loss_iou = 1.0 - iou

        # Center distance squared
        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2
        rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

        # Smallest enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + self.eps

        # Distance ratio with detached enclosing box diagonal to prevent gradient blockage
        dist_ratio = torch.exp(rho2 / c2.detach())

        # WIoU v1
        loss_wiou_v1 = dist_ratio * loss_iou

        if self.version == 1:
            return loss_wiou_v1.squeeze(-1)

        # Update running mean of IoU loss during training
        with torch.no_grad():
            batch_mean = loss_iou.mean()
            if self.training and not torch.isnan(batch_mean):
                self.iou_mean = self.momentum * self.iou_mean + (1.0 - self.momentum) * batch_mean

        # Outlier degree beta
        beta = (loss_iou.detach() / (self.iou_mean + self.eps)).clamp(min=self.eps)

        if self.version == 2:
            # Monotonic focusing factor
            loss_wiou_v2 = (beta ** 0.5) * loss_wiou_v1
            return loss_wiou_v2.squeeze(-1)

        elif self.version == 3:
            # Non-monotonic focusing factor r
            # Downweights high-quality samples and de-emphasizes extreme outliers (noisy annotations)
            r = beta / (self.delta * (self.alpha ** (beta - self.delta)))
            loss_wiou_v3 = r * loss_wiou_v1
            return loss_wiou_v3.squeeze(-1)


def compute_bbox_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    loss_type: str = 'wiou_v3'
) -> torch.Tensor:
    """
    Flexible wrapper supporting multiple loss functions for ablation study:
    - 'ciou': Standard Complete IoU in YOLOv8
    - 'wiou_v1': Wise-IoU version 1
    - 'wiou_v2': Wise-IoU version 2
    - 'wiou_v3': Wise-IoU version 3 (Proposed)
    - 'siou': SCYLLA-IoU
    - 'diou': Distance-IoU
    """
    if loss_type == 'wiou_v3':
        criterion = WiseIoULoss(version=3)
        return criterion(pred_boxes, target_boxes)
    elif loss_type == 'wiou_v1':
        criterion = WiseIoULoss(version=1)
        return criterion(pred_boxes, target_boxes)
    elif loss_type == 'wiou_v2':
        criterion = WiseIoULoss(version=2)
        return criterion(pred_boxes, target_boxes)
    else:
        # Fallback to standard IoU loss
        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        area2 = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union = area1 + area2 - inter + 1e-7
        iou = inter / union
        return 1.0 - iou
