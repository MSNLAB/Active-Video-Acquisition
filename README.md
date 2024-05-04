## <div align="center">Multi-Edge Reinforced Collaborative Data Acquisition for Continuous Video Analytics by Prioritizing Quality over Quantity<div> 

<div align="center">


![](https://github.com/MSNLAB/Active-Video-Acquisition/actions/workflows/lint.yml/badge.svg)
![](https://github.com/MSNLAB/Active-Video-Acquisition/actions/workflows/test.yml/badge.svg)
![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)

<!--
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/)
![AAAI'25](https://img.shields.io/badge/AAAI'25-155D6D)
-->

</div>

## 🌈 Overview

> In this work, we propose a multi-edge collaborative active video acquisition (AVA) framework to collaboratively learn a reinforced video acquisition strategy to identify informative video frames from multiple edge nodes that best enhance model accuracy, avoiding redundancy across edges. Extensive experiments on three video datasets demonstrate that, our method achieves comparable performance to full-set video training while utilizing only 20% of the data in classification tasks. In object detection tasks, our methods can maintain productive accuracy with a reduction of nearly 70% in training video frames.

![architecture](https://github.com/user-attachments/assets/ee50ee50-06e5-4ea3-808f-ad3880f1fa28)

## 🚀 Quick start

1. Please install all dependencies from `requirements.txt` after creating your own python environment with `python >= 3.10`.

```bash
pip3 install -r requirements.txt
```

2. Download the datasets, and then unzip on the project root path.

- MOT15: https://motchallenge.net/results/MOT15/
- CORe50: https://vlomonaco.github.io/core50/
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset

3. Startup the experiments.

```bash
python3 train_policy.py \
    --dataset "visdrone" \
    --source_root "/path/to/visdrone"
```

## 🛠️ Configuration

You can configure the training process by passing arguments to the script. Here are some of the key configurations:

- `--dataset`: Specify the dataset to use (e.g., visdrone_n3, core50_ni_c10, mot15_n3).
- `--model`: Choose the model architecture (e.g., fasterrcnn_resnet50_fpn, resnet50, vit_base).
- `--agent_net`: Define the network type for the agent (e.g., transformer).
- `--episodes`: Set the number of training episodes.
- `--env_task`: Choose the task type (e.g., classification or detection).
- `--node_cnt`: Number of nodes (agents) for distributed training.
- `--checkpoint_interval`: Save a checkpoint every `N` episodes.

## 📦 Datasets

| Dataset              | Task         | Description                                       |
|----------------------|--------------|---------------------------------------------------|
| `core50_ni_c10`      | Classification | CORE50 dataset with 10 classes, scenario NI       |
| `core50_ni_c50`      | Classification | CORE50 dataset with 50 classes, scenario NI       |
| `core50_simulation`  | Classification | Simulated CORE50 dataset                          |
| `visdrone_n1`        | Detection    | VisDrone dataset with 1 node configuration       |
| `visdrone_n3`        | Detection    | VisDrone dataset with 3 nodes configuration      |
| `visdrone_n5`        | Detection    | VisDrone dataset with 5 nodes configuration      |
| `visdrone_n8`        | Detection    | VisDrone dataset with 8 nodes configuration      |
| `mot15_n1`           | Detection    | MOT15 dataset with 1 node configuration          |
| `mot15_n3`           | Detection    | MOT15 dataset with 3 nodes configuration         |
| `mot15_n5`           | Detection    | MOT15 dataset with 5 nodes configuration         |
| `mot15_n8`           | Detection    | MOT15 dataset with 8 nodes configuration         |

## ✨ Models

| Model                           | Task         | Description                                       |
|---------------------------------|--------------|---------------------------------------------------|
| `resnet18`                      | Classification | ResNet-18 model                                  |
| `resnet34`                      | Classification | ResNet-34 model                                  |
| `resnet50`                      | Classification | ResNet-50 model                                  |
| `resnet101`                     | Classification | ResNet-101 model                                 |
| `resnet152`                     | Classification | ResNet-152 model                                 |
| `vgg11`                         | Classification | VGG-11 model                                     |
| `vgg13`                         | Classification | VGG-13 model                                     |
| `vgg16`                         | Classification | VGG-16 model                                     |
| `vgg19`                         | Classification | VGG-19 model                                     |
| `vit_tiny`                      | Classification | Vision Transformer Tiny model                    |
| `vit_small`                     | Classification | Vision Transformer Small model                   |
| `vit_base`                      | Classification | Vision Transformer Base model                    |
| `vit_large`                     | Classification | Vision Transformer Large model                   |
| `swin_tiny`                     | Classification | Swin Transformer Tiny model                      |
| `swin_small`                    | Classification | Swin Transformer Small model                     |
| `swin_base`                     | Classification | Swin Transformer Base model                      |
| `swin_large`                    | Classification | Swin Transformer Large model                     |
| `cnn`                           | Classification | Custom CNN model                                 |
| `fasterrcnn_resnet50_fpn`       | Detection    | Faster R-CNN with ResNet-50 FPN backbone         |
| `fasterrcnn_resnet50_fpn_v2`    | Detection    | Faster R-CNN with ResNet-50 FPN v2 backbone      |
| `fasterrcnn_mobilenet_v3_large_fpn`| Detection| Faster R-CNN with MobileNet v3 large FPN backbone|
| `fasterrcnn_mobilenet_v3_large_320_fpn`| Detection| Faster R-CNN with MobileNet v3 large 320 FPN backbone|


## 🧩 Multi-GPU Training

To start training, run the following command:

```bash
python3 train_policy.py --dataset visdrone_n3 --model fasterrcnn_resnet50_fpn --episodes 100000
```

For distributed training across multiple GPUs, use the `--parallel` flag to specify the number of GPUs:

```bash
python3 train_policy.py --parallel <number_of_gpus>
```

## 📖 Citation

If you use this for research, please cite. The example BibTeX entry will be given after paper review.


## 📚 Acknowledgments

The codes are expanded on the following projects:

- [Active-Learning-as-a-Service (ALaaS)](https://github.com/MLSysOps/Active-Learning-as-a-Service)
- [Video-Platform-as-a-Service (VPaaS)](https://arxiv.org/abs/2102.03012)
- [Pytorch-VSUMM-Reinforce (VSUMM)](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)
- [ActVideo: Active Continuous Learning for Video Analytics](https://github.com/MSNLAB/ActVideo?tab=readme-ov-file)

## 📝 License

The project is available as open source under the terms of the [MIT License](https://opensource.org/license/mit).
