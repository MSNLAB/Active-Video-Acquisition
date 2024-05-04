import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import default_collate


class ClassificationDataset(Dataset):
    _default_transforms = T.Compose(
        (
            T.ToTensor(),
            T.Resize(128),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
    )

    def __init__(self, imgs, labels, transforms=None):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms

        if transforms is None:
            self.transforms = self._default_transforms

    def __getitem__(self, index):
        data = self.transforms(self.imgs[index])
        target = self.labels[index]

        return data, target

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        return default_collate(batch)


class DetectionDataset(Dataset):
    _default_transforms = A.Compose(
        transforms=[
            A.Resize(640, 960),  # (277, 830), (640, 960)
            A.Normalize(),
            A.ToFloat(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    def __init__(self, frames, labels, transforms=None):
        self.frames = frames
        self.labels = labels
        self.transforms = transforms

        if transforms is None:
            self.transforms = self._default_transforms

    def __getitem__(self, index):
        data = self.transforms(
            image=np.array(Image.open(self.frames[index]).convert("RGB")),
            bboxes=self.labels[index]["boxes"],
            labels=self.labels[index]["labels"],
        )
        frame = data["image"]
        label = {
            "boxes": torch.tensor(data["bboxes"], dtype=torch.float32),
            "labels": torch.tensor(data["labels"], dtype=torch.int64),
        }
        return frame, label

    def __len__(self):
        return len(self.frames)

    def collate_fn(self, batch):
        return tuple(zip(*batch))
