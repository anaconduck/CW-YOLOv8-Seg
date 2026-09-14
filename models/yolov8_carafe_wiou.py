"""YOLOv8 integration with CARAFE and Wise-IoU loss."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.carafe_module import CARAFE
from models.wiou_loss import WiseIoULoss

try:
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    setattr(modules, 'CARAFE', CARAFE)
    setattr(tasks, 'CARAFE', CARAFE)
    if hasattr(modules, '__all__') and 'CARAFE' not in modules.__all__:
        modules.__all__.append('CARAFE')
except ImportError:
    pass


def register_carafe_to_ultralytics():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    setattr(modules, 'CARAFE', CARAFE)
    setattr(tasks, 'CARAFE', CARAFE)


def patch_ultralytics_loss_with_wiou(version: int = 3, alpha: float = 1.9, delta: float = 3.0):
    from ultralytics.utils.loss import BboxLoss

    wiou_criterion = WiseIoULoss(version=version, alpha=alpha, delta=delta)

    def wiou_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou_loss = wiou_criterion(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = (iou_loss.unsqueeze(-1) * weight).sum() / target_scores_sum

        if self.dfl:
            target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    BboxLoss.forward = wiou_forward
    print(f"[INFO] BboxLoss patched with Wise-IoU v{version}")


def get_model(yaml_config_path: str = None, weights_path: str = None):
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

