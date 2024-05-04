import torch.nn as nn
from torchvision.models.vgg import *
from torchvision.ops import FrozenBatchNorm2d
import torch.nn.functional as F

from utils import set_module_requires_grad


def vgg_finetune_parameter_groups(model):
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


class VGG(nn.Module):
    def __init__(self, net, num_classes=1000, pretrain=True):
        super(VGG, self).__init__()
        if net is None:
            net = "vgg16"

        if net == "vgg11":
            self.backbone = vgg11(pretrain)
        elif net == "vgg13":
            self.backbone = vgg13(pretrain)
        elif net == "vgg16":
            self.backbone = vgg16(pretrain)
        elif net == "vgg19":
            self.backbone = vgg19(pretrain)
        else:
            raise NotImplementedError(f"given vgg {net} is not supported.")

        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
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
