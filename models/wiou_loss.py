"""Wise-IoU (WIoU) loss (Tong et al., 2023)."""

import math
import torch
import torch.nn as nn


class WiseIoULoss(nn.Module):
    def __init__(
        self,
        version: int = 3,
        alpha: float = 1.9,
        delta: float = 3.0,
        momentum: float = 0.9,
        eps: float = 1e-7
    ):
        super().__init__()
        assert version in [1, 2, 3], f"Invalid WIoU version: {version}"
        self.version = version
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.chunk(4, -1)

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        b1_area = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        b2_area = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        union_area = b1_area + b2_area - inter_area + self.eps

        iou = (inter_area / union_area).clamp(min=0, max=1.0)
        loss_iou = 1.0 - iou

        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2
        rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + self.eps

        dist_ratio = torch.exp(rho2 / c2.detach())
        loss_wiou_v1 = dist_ratio * loss_iou

        if self.version == 1:
            return loss_wiou_v1.squeeze(-1)

        with torch.no_grad():
            batch_mean = loss_iou.mean()
            if self.training and not torch.isnan(batch_mean):
                self.iou_mean = self.momentum * self.iou_mean + (1.0 - self.momentum) * batch_mean

        beta = (loss_iou.detach() / (self.iou_mean + self.eps)).clamp(min=self.eps)

        if self.version == 2:
            loss_wiou_v2 = (beta ** 0.5) * loss_wiou_v1
            return loss_wiou_v2.squeeze(-1)

        elif self.version == 3:
            r = beta / (self.delta * (self.alpha ** (beta - self.delta)))
            loss_wiou_v3 = r * loss_wiou_v1
            return loss_wiou_v3.squeeze(-1)


def compute_bbox_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    loss_type: str = 'wiou_v3'
) -> torch.Tensor:
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

