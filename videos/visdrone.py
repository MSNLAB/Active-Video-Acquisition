import os

import numpy as np
import pandas as pd

from videos.common import DetectionDataset

__all__ = (
    "visdrone_n1",
    "visdrone_n3",
    "visdrone_n5",
    "visdrone_n8",
)

annotation_cols = (
    "frame_index",
    "target_id",
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
    "score",
    "object_category",
    "truncation",
    "occlusion",
)

object_category = (
    "ignored regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
)

sequences = (
    "uav0000009_03358_v",
    "uav0000013_00000_v",
    "uav0000013_01073_v",
    "uav0000013_01392_v",
    "uav0000020_00406_v",
    "uav0000071_03240_v",
    "uav0000072_04488_v",
    "uav0000072_05448_v",
    "uav0000072_06432_v",
    "uav0000073_00600_v",
    "uav0000073_04464_v",
    "uav0000076_00720_v",
    "uav0000077_00720_v",
    "uav0000079_00480_v",
    "uav0000084_00000_v",
    "uav0000086_00000_v",
    "uav0000088_00290_v",
    "uav0000099_02109_v",
    "uav0000117_02622_v",
    "uav0000119_02301_v",
    "uav0000120_04775_v",
    "uav0000124_00944_v",
    "uav0000126_00001_v",
    "uav0000137_00458_v",
    "uav0000138_00000_v",
    "uav0000140_01590_v",
    "uav0000143_02250_v",
    "uav0000145_00000_v",
    "uav0000150_02310_v",
    "uav0000161_00000_v",
    "uav0000182_00000_v",
    "uav0000188_00000_v",
    "uav0000201_00000_v",
    "uav0000218_00001_v",
    "uav0000222_03150_v",
    "uav0000239_03720_v",
    "uav0000239_12336_v",
    "uav0000243_00001_v",
    "uav0000244_01440_v",
    "uav0000248_00001_v",
    "uav0000249_00001_v",
    "uav0000249_02688_v",
    "uav0000263_03289_v",
    "uav0000264_02760_v",
    "uav0000266_03598_v",
    "uav0000266_04830_v",
    "uav0000268_05773_v",
    "uav0000270_00001_v",
    "uav0000273_00001_v",
    "uav0000278_00001_v",
    "uav0000279_00001_v",
    "uav0000281_00460_v",
    "uav0000288_00001_v",
    "uav0000289_00001_v",
    "uav0000289_06922_v",
    "uav0000295_02300_v",
    "uav0000297_00000_v",
    "uav0000297_02761_v",
    "uav0000300_00000_v",
    "uav0000305_00000_v",
    "uav0000306_00230_v",
    "uav0000307_00000_v",
    "uav0000308_00000_v",
    "uav0000308_01380_v",
    "uav0000309_00000_v",
    "uav0000315_00000_v",
    "uav0000316_01288_v",
    "uav0000323_01173_v",
    "uav0000326_01035_v",
    "uav0000329_04715_v",
    "uav0000339_00001_v",
    "uav0000342_04692_v",
    "uav0000352_05980_v",
    "uav0000355_00001_v",
    "uav0000357_00920_v",
    "uav0000360_00001_v",
    "uav0000361_02323_v",
    "uav0000363_00001_v",
    "uav0000366_00001_v",
    "uav0000370_00001_v",
)


def collect_frames(root, sequences):
    if not isinstance(sequences, (list, tuple)):
        sequences = (sequences,)

    frames = []
    targets = []

    for sequence in sequences:
        frame_path = os.path.join(root, "sequences", sequence)
        frame_names = sorted(list(os.listdir(frame_path)))

        annotation_path = os.path.join(root, "annotations", sequence + ".txt")
        annotations = pd.read_csv(annotation_path, header=None, names=annotation_cols)

        for frame_name in frame_names:
            _id = int(frame_name.split(".")[0])
            _path = os.path.join(frame_path, frame_name)
            _labels = annotations[annotations["frame_index"] == _id]

            boxes = []
            labels = []
            for _, _label in _labels.iterrows():
                label = _label["object_category"]
                if label != 0:
                    x_min = int(_label["bbox_left"])
                    y_min = int(_label["bbox_top"])
                    x_max = x_min + int(_label["bbox_width"]) - 2
                    y_max = y_min + int(_label["bbox_height"]) - 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)

            if len(boxes) != 0 and len(labels) != 0:
                frames.append(_path)
                targets.append({"boxes": boxes, "labels": labels})

    return frames, targets


def visdrone(root="./videos/visdrone/", node_cnt=1):
    tr_data = []

    val_frames, val_labels = [], []

    for nid in range(node_cnt):
        frames, targets = collect_frames(root, sequences[nid])

        val_rand_pick = np.random.choice(range(len(frames)), size=len(frames) // 5, replace=False)

        tr_frames, tr_labels = [], []
        for idx, (frame, target) in enumerate(zip(frames, targets)):
            if idx in val_rand_pick:
                val_frames.append(frame)
                val_labels.append(target)
            else:
                tr_frames.append(frame)
                tr_labels.append(target)

        tr_data.append(DetectionDataset(tr_frames, tr_labels))

    val_data = DetectionDataset(val_frames, val_labels)

    return tr_data, val_data


def visdrone_n1(root="./videos/visdrone/"):
    return visdrone(root, 1)


def visdrone_n3(root="./videos/visdrone/"):
    return visdrone(root, 3)


def visdrone_n5(root="./videos/visdrone/"):
    return visdrone(root, 5)


def visdrone_n8(root="./videos/visdrone/"):
    return visdrone(root, 8)
