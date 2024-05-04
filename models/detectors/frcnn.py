import torch.nn as nn
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

from utils import set_module_requires_grad


def frcnn_finetune_parameter_groups(model):
    set_module_requires_grad(model, requires_grad=False)
    set_module_requires_grad(model.rpn, requires_grad=True)
    set_module_requires_grad(model.roi_heads, requires_grad=True)
    parameter_groups = (
        {"params": [p for p in model.rpn.head.conv.parameters()], "lr": 1e-5},
        {"params": [p for p in model.rpn.head.bbox_pred.parameters()], "lr": 5e-4},
        {"params": [p for p in model.rpn.head.cls_logits.parameters()], "lr": 5e-4},
        {"params": [p for p in model.roi_heads.box_head.fc6.parameters()], "lr": 1e-5},
        {"params": [p for p in model.roi_heads.box_head.fc7.parameters()], "lr": 5e-4},
        {"params": [p for p in model.roi_heads.box_predictor.cls_score.parameters()], "lr": 5e-3},
        {"params": [p for p in model.roi_heads.box_predictor.bbox_pred.parameters()], "lr": 5e-3},
    )
    return parameter_groups


class FRCNN(nn.Module):
    def __init__(self, net=None, num_classes=91, pretrain=True):
        super(FRCNN, self).__init__()
        if net is None:
            net = "fasterrcnn_resnet50_fpn"

        kwargs_factory = {"pretrained_backbone": True, "trainable_backbone_layers": 0}

        if net == "fasterrcnn_resnet50_fpn":
            frcnn = fasterrcnn_resnet50_fpn(pretrain, **kwargs_factory)
        elif net == "fasterrcnn_resnet50_fpn_v2":
            frcnn = fasterrcnn_resnet50_fpn_v2(pretrain, **kwargs_factory)
        elif net == "fasterrcnn_mobilenet_v3_large_fpn":
            frcnn = fasterrcnn_mobilenet_v3_large_fpn(pretrain, **kwargs_factory)
        elif net == "fasterrcnn_mobilenet_v3_large_320_fpn":
            frcnn = fasterrcnn_mobilenet_v3_large_320_fpn(pretrain, **kwargs_factory)
        else:
            raise NotImplementedError(f"given faster rcnn {net} is not supported.")

        self.transform = frcnn.transform
        self.backbone = frcnn.backbone
        self.rpn = frcnn.rpn
        self.roi_heads = frcnn.roi_heads

        self.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes)

    def forward(self, images, targets=None):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {
            **detector_losses,
            **proposal_losses,
        }

        if self.training:
            return losses

        return detections
