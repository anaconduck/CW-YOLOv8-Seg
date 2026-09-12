"""
Integration wrapper for YOLOv8 with CARAFE and Wise-IoU Loss.
Registers CARAFE into Ultralytics module registry and injects WIoU loss into the segmentation trainer.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.carafe_module import CARAFE
from models.wiou_loss import WiseIoULoss

# Register CARAFE into Ultralytics modules
try:
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    # Expose CARAFE so parse_model can construct it from YAML
    setattr(modules, 'CARAFE', CARAFE)
    setattr(tasks, 'CARAFE', CARAFE)
    if hasattr(modules, '__all__') and 'CARAFE' not in modules.__all__:
        modules.__all__.append('CARAFE')
except ImportError:
    pass


def register_carafe_to_ultralytics():
    """Ensures CARAFE is recognized by Ultralytics YAML parser."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    setattr(modules, 'CARAFE', CARAFE)
    setattr(tasks, 'CARAFE', CARAFE)


def patch_ultralytics_loss_with_wiou(version: int = 3, alpha: float = 1.9, delta: float = 3.0):
    """
    Patches Ultralytics BboxLoss / SegmentationLoss with Wise-IoU Loss.
    Args:
        version (int): WIoU version (1, 2, or 3). Default is 3.
        alpha (float): Hyperparameter alpha for WIoU v3 non-monotonic mapping.
        delta (float): Outlier threshold delta for WIoU v3.
    """
    from ultralytics.utils.loss import BboxLoss

    wiou_criterion = WiseIoULoss(version=version, alpha=alpha, delta=delta)

    original_forward = BboxLoss.forward

    def wiou_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Modified BboxLoss forward using Wise-IoU."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # Calculate WIoU between predicted boxes and ground truth boxes
        iou_loss = wiou_criterion(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = (iou_loss.unsqueeze(-1) * weight).sum() / target_scores_sum

        # DFL (Distribution Focal Loss) component from original YOLOv8
        if self.dfl:
            target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    # Monkey patch BboxLoss forward
    BboxLoss.forward = wiou_forward
    print(f"[INFO] Successfully patched Ultralytics BboxLoss with Wise-IoU v{version} (alpha={alpha}, delta={delta})")


def get_model(yaml_config_path: str = None, weights_path: str = None):
    """
    Factory function to initialize YOLOv8-CARAFE model.
    """
    register_carafe_to_ultralytics()
    from ultralytics import YOLO

    if yaml_config_path:
        model = YOLO(yaml_config_path)
    elif weights_path:
        model = YOLO(weights_path)
    else:
        default_cfg = PROJECT_ROOT / "models" / "configs" / "yolov8s-seg-carafe.yaml"
        model = YOLO(str(default_cfg))

    return model
