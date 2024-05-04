from typing import Any

import torch
from torch.utils.data import DataLoader

from models.classifiers import ResNet, VGG, ViT
from models.detectors import FRCNN
from utils import *


class ResNetFeatureExtractor:
    def __init__(self, net=None, device=None):
        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        self.backbone = ResNet(net).backbone

    @torch.no_grad()
    def embedding(self, images):
        return self.backbone(images)

    @torch.no_grad()
    def __call__(self, dataset, batch_size=128):
        features = []
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        for images, targets in dataloader:
            images, targets = to_device((images, targets), self.device)
            features.append(self.embedding(images).detach().cpu())
        self.backbone.cpu()

        features = torch.cat(features, dim=0)
        features = features.view(features.shape[0], -1)
        return features


class VGGFeatureExtractor:
    def __init__(self, net=None, device=None):
        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        self.backbone = VGG(net).backbone

    @torch.no_grad()
    def embedding(self, images):
        return self.backbone(images)

    @torch.no_grad()
    def __call__(self, dataset, batch_size=128):
        features = []
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        for images, targets in dataloader:
            images, targets = to_device((images, targets), self.device)
            features.append(self.embedding(images).detach().cpu())
        self.backbone.cpu()

        features = torch.cat(features, dim=0)
        features = features.view(features.shape[0], -1)
        return features


class ViTFeatureExtractor:
    def __init__(self, net=None, device=None):
        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        self.backbone = ViT(net).backbone

    @torch.no_grad()
    def embedding(self, images):
        return self.backbone(images)

    @torch.no_grad()
    def __call__(self, dataset, batch_size=128):
        features = []
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        for images, targets in dataloader:
            images, targets = to_device((images, targets), self.device)
            features.append(self.embedding(images).detach().cpu())
        self.backbone.cpu()

        features = torch.cat(features, dim=0)
        features = features.view(features.shape[0], -1)
        return features


class FRCNNFeatureExtractor:
    def __init__(self, net=None, device=None):
        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        self.frcnn = FRCNN(net)

    @torch.no_grad()
    def embedding(self, images, targets=None):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.frcnn.transform(images, targets)
        features = self.frcnn.backbone(images.tensors)
        return features

    @torch.no_grad()
    def __call__(self, dataset, batch_size=8):
        features = []
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )
        self.frcnn.to(self.device)
        self.frcnn.eval()
        for images, targets in dataloader:
            images, targets = to_device((images, targets), self.device)
            features.append(self.embedding(images, targets)["pool"].detach().cpu())
        self.frcnn.cpu()

        features = torch.cat(features, dim=0)
        features = features.view(features.shape[0], -1)
        return features
