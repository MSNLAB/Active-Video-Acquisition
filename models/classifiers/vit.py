import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.ops import FrozenBatchNorm2d

import timm

from utils import set_module_requires_grad


def vit_finetune_parameter_groups(model):
    set_module_requires_grad(model, requires_grad=False)
    set_module_requires_grad(model.classifier, requires_grad=True)

    # replace BatchNorm2d with FrozenBatchNorm2d
    modules = {n: m for n, m in model.backbone.named_modules()}
    for name, module in modules.items():
        if isinstance(module, nn.BatchNorm2d):
            frozen_module = FrozenBatchNorm2d(module.num_features)
            frozen_module.weight.data = module.weight.data.clone().detach()
            frozen_module.bias.data = module.bias.data.clone().detach()
            frozen_module.running_mean.data = module.running_mean.data.clone().detach()
            frozen_module.running_var.data = module.running_var.data.clone().detach()

            name = name.split(".")
            module = model.backbone
            for n in name[:-1]:
                module = getattr(module, n)
            setattr(module, name[-1], frozen_module)

    parameter_groups = ({"params": [p for p in model.classifier.parameters()], "lr": 1e-3},)

    return parameter_groups


class ViT(nn.Module):
    def __init__(self, net, num_classes=1000, pretrain=True):
        super(ViT, self).__init__()
        if net is None:
            net = "vit_tiny"

        if net == "vit_tiny":
            self.backbone = timm.create_model("vit_tiny_patch16_224", pretrain)
            in_features = 192
        elif net == "vit_small":
            self.backbone = timm.create_model("vit_small_patch16_224", pretrain)
            in_features = 384
        elif net == "vit_base":
            self.backbone = timm.create_model("vit_base_patch16_224", pretrain)
            in_features = 768
        elif net == "vit_large":
            self.backbone = timm.create_model("vit_large_patch16_224", pretrain)
            in_features = 1024
        elif net == "swin_tiny":
            self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrain)
            in_features = 768
        elif net == "swin_small":
            self.backbone = timm.create_model("swin_small_patch4_window7_224", pretrain)
            in_features = 768
        elif net == "swin_base":
            self.backbone = timm.create_model("swin_base_patch4_window7_224", pretrain)
            in_features = 1024
        elif net == "swin_large":
            self.backbone = timm.create_model("swin_large_patch4_window7_224", pretrain)
            in_features = 1536
        else:
            raise NotImplementedError(f"given vision transformer {net} is not supported.")

        if net in ("vit_tiny", "vit_small", "vit_base", "vit_large"):
            self.backbone.head = nn.Identity()
        elif net in ("swin_tiny", "swin_small", "swin_base", "swin_large"):
            self.backbone.head.fc = nn.Identity()
        else:
            raise NotImplementedError(f"given vision transformer {net} is not supported.")

        self.backbone = nn.Sequential(
            T.Resize([224, 224]),
            self.backbone,
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, y):
        pred = self.classifier(self.backbone(x))

        if self.training:
            return F.cross_entropy(pred, y)

        return pred
