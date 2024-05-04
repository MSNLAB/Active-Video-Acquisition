import os

import numpy as np
import pandas as pd
import configparser

from videos.common import DetectionDataset

__all__ = (
    "mot15_n1",
    "mot15_n3",
    "mot15_n5",
    "mot15_n8",
)

annotation_cols = (
    "frame_index",
    "target_id",
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
    "score",
    "3d-x",
    "3d-y",
    "3d-z",
)

sequences = (
    "KITTI-13",
    "KITTI-16",
    "KITTI-17",
    "KITTI-19",
    "ETH-Sunnyday",
    "ETH-Crossing",
    "ETH-Jelmoli",
    "ETH-Pedcross2",
    "ETH-Bahnhof",
    "ETH-Linthescher",
    "ADL-Rundle-1",
    "ADL-Rundle-3",
    "ADL-Rundle-6",
    "ADL-Rundle-8",
    "AVG-TownCentre",
    "PETS09-S2L1",
    "PETS09-S2L2",
    "TUD-Campus",
    "TUD-Crossing",
    "TUD-Stadtmitte",
    "Venice-1",
    "Venice-2",
)


def collect_frames(root, sequences):
    if not isinstance(sequences, (list, tuple)):
        sequences = (sequences,)

    frames = []
    targets = []

    for sequence in sequences:
        seq_path = os.path.join(root, sequence)

        config = configparser.ConfigParser()
        config.read(os.path.join(seq_path, "seqinfo.ini"))

        frame_path = os.path.join(root, sequence, config["Sequence"]["imDir"])
        frame_names = sorted(list(os.listdir(frame_path)))

        annotation_path = os.path.join(root, sequence, "det/det.txt")
        annotations = pd.read_csv(annotation_path, header=None, names=annotation_cols)

        for frame_name in frame_names:
            _id = int(frame_name.split(".")[0])
            _path = os.path.join(frame_path, frame_name)
            _labels = annotations[annotations["frame_index"] == _id]

            boxes = []
            labels = []
            for _, _label in _labels.iterrows():
                label = 1
                if label != 0:
                    x_min = int(_label["bbox_left"])
                    y_min = int(_label["bbox_top"])
                    x_max = x_min + int(_label["bbox_width"]) - 2
                    y_max = y_min + int(_label["bbox_height"]) - 2
                    boxes.append([max(x_min, 0), max(y_min, 0),
                                  min(x_max, int(config["Sequence"]["imWidth"])),
                                  min(y_max, int(config["Sequence"]["imHeight"]))])
                    labels.append(label)

            if len(boxes) != 0 and len(labels) != 0:
                frames.append(_path)
                targets.append({"boxes": boxes, "labels": labels})

    return frames, targets


def mot15(root="./videos/mot15/", node_cnt=1):
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


def mot15_n1(root="./videos/mot15/"):
    return mot15(root, 1)


def mot15_n3(root="./videos/mot15/"):
    return mot15(root, 3)


def mot15_n5(root="./videos/mot15/"):
    return mot15(root, 5)


def mot15_n8(root="./videos/mot15/"):
    return mot15(root, 8)
