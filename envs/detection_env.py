import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from envs.utils.feature_compression import pca_compression
from envs.utils.feature_extractor import FRCNNFeatureExtractor
from envs.utils.federated_learning import aggregate_params
from envs.trainers import FRCNNTrainer
from models.detectors.frcnn import frcnn_finetune_parameter_groups
from utils import *
from videos.common import DetectionDataset

logger = get_logger(__file__)


class DetectionTrainingEnv:
    def __init__(
        self,
        node_cnt: int,
        window_size: int,
        detector: nn.Module,
        tr_data: List[DetectionDataset],
        val_data: DetectionDataset,
        compressed_rep_dim: int = 32,
        lambda_div_reward: float = 0.1,
        lambda_rep_reward: float = 0.2,
        lambda_eff_reward: float = 0.5,
        lambda_acc_reward: float = 1.0,
        device: str = None,
        verbose: bool = False,
    ):
        self.node_cnt = node_cnt
        self.window_size = window_size
        self.detector = detector

        self.tr_data = tr_data
        self.val_data = val_data

        self.compressed_rep_dim = compressed_rep_dim
        self.lambda_div_reward = lambda_div_reward
        self.lambda_rep_reward = lambda_rep_reward
        self.lambda_eff_reward = lambda_eff_reward
        self.lambda_acc_reward = lambda_acc_reward

        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        self.verbose = verbose

        self.reps = self._pre_load_reps()

    @cache_empty
    def _pre_load_reps(self):
        self.reps = []
        extractor = FRCNNFeatureExtractor(device=self.device)
        for nid in range(self.node_cnt):
            rep = extractor(self.tr_data[nid])
            rep = pca_compression(rep, self.compressed_rep_dim)
            self.reps.append(rep)
        return self.reps

    def reset(self):
        trainers = [] if not hasattr(self, "trainers") else self.trainers
        states = [] if not hasattr(self, "states") else self.states

        if len(trainers) == 0:
            for nid in range(self.node_cnt):
                net = copy.deepcopy(self.detector)
                param_groups = frcnn_finetune_parameter_groups(net)
                trainer = FRCNNTrainer(
                    model=net,
                    tr_dataset=self.tr_data[nid],
                    val_dataset=self.val_data,
                    verbose=self.verbose,
                    parameter_groups=param_groups,
                )
                trainers.append(trainer)
        else:
            for trainer in trainers:
                trainer.reset_params()

        if len(states) == 0:
            for nid in range(self.node_cnt):
                state = []
                for offset_start in range(0, len(self.reps[nid]), self.window_size):
                    offset_end = offset_start + self.window_size
                    state.append(self.reps[nid][offset_start:offset_end, :])
                states.append(state)

        self.trainers = trainers
        self.states = states
        return states

    def step(self, actions, log_probs):
        picks = []
        reactions = {
            "rewards": [[] for _ in range(self.node_cnt)],
            "div_rewards": [[] for _ in range(self.node_cnt)],
            "rep_rewards": [[] for _ in range(self.node_cnt)],
            "eff_rewards": [[] for _ in range(self.node_cnt)],
            "acc_rewards": [[] for _ in range(self.node_cnt)],
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "map_small": 0.0,
            "map_medium": 0.0,
            "map_large": 0.0,
            "mar_1": 0.0,
            "mar_10": 0.0,
            "mar_100": 0.0,
            "mar_small": 0.0,
            "mar_medium": 0.0,
            "mar_large": 0.0,
        }

        for nid in range(self.node_cnt):
            n_picks = []

            for sid in range(len(self.states[nid])):
                action = actions[nid][sid]

                pick_ids = torch.where(action == 1)[0]
                pick_offsets = sid * self.window_size + pick_ids

                reps = self.reps[nid][sid * self.window_size : (sid + 1) * self.window_size, :]

                div_reward = self._compute_div_reward(reps, pick_ids)
                rep_reward = self._compute_rep_reward(reps, pick_ids)

                reactions["div_rewards"][nid].append(div_reward)
                reactions["rep_rewards"][nid].append(rep_reward)

                n_picks.append(pick_offsets)

            n_picks = torch.cat(n_picks, dim=0)

            picks.append(n_picks)

        # collaborative learning
        acc = {}
        for comm in range(1, 101):
            for trainer, pick in zip(self.trainers, picks):
                trainer.step(epoch_cnt=1, select_ids=pick)

            params = aggregate_params(
                models=[trainer.model for trainer in self.trainers],
                weights=[len(pick) for pick in picks],
            )

            for trainer in self.trainers:
                trainer.reset_params(params)

            if comm >= 98:
                epoch_acc = self.trainers[0].eval()
                for tag, value in epoch_acc.items():
                    if tag not in acc.keys():
                        acc[tag] = []
                    acc[tag].append(value)

        acc = {k: np.mean(v) for k, v in acc.items()}
        eff = self._compute_eff_reward(
            sum([len(pick) for pick in picks]) /
            sum([len(rep) for rep in self.reps])
        )
        for nid in range(self.node_cnt):
            for sid in range(len(self.states[nid])):
                reactions["eff_rewards"][nid].append(eff)
                reactions["acc_rewards"][nid].append(acc["map"] + 1.0 * acc["map/large"] + 1.0 * acc["mar/large"])
                reactions["rewards"][nid].append(
                    self.lambda_acc_reward * reactions["acc_rewards"][nid][sid]
                    + self.lambda_eff_reward * reactions["eff_rewards"][nid][sid]
                    + self.lambda_div_reward * reactions["div_rewards"][nid][sid]
                    + self.lambda_rep_reward * reactions["rep_rewards"][nid][sid]
                )

        reactions["map"] += acc["map"]
        reactions["map_50"] += acc["map/50"]
        reactions["map_75"] += acc["map/75"]
        reactions["map_small"] += acc["map/small"]
        reactions["map_medium"] += acc["map/medium"]
        reactions["map_large"] += acc["map/large"]
        reactions["mar_1"] += acc["mar/1"]
        reactions["mar_10"] += acc["mar/10"]
        reactions["mar_100"] += acc["mar/100"]
        reactions["mar_small"] += acc["mar/small"]
        reactions["mar_medium"] += acc["mar/medium"]
        reactions["mar_large"] += acc["mar/large"]

        return reactions

    @torch.no_grad()
    def _compute_div_reward(self, reps, pick_ids):
        device = self.device

        div_reward = -0.5 * torch.ones([]).to(device)
        pick_reps = reps[pick_ids, :].to(device)

        if len(pick_reps) > 1:
            norm_s = pick_reps / pick_reps.norm(p=2, dim=1, keepdim=True)
            div_reward = (1.0 - norm_s @ norm_s.t()).sum()
            div_reward /= (norm_s.shape[0] - 1) * norm_s.shape[0]

        return div_reward.cpu()

    @torch.no_grad()
    def _compute_rep_reward(self, reps, pick_ids):
        device = self.device

        rep_reward = -0.5 * torch.ones([]).to(device)
        reps = reps.to(device)
        pick_ids = pick_ids.to(device)

        if len(pick_ids) > 0:
            norm_s = reps / reps.norm(p=2, dim=1, keepdim=True)
            dist_mat = (1.0 - norm_s @ norm_s.t())[:, pick_ids]
            dist_mat = dist_mat.min(1, keepdim=True)[0]
            rep_reward = torch.exp(-dist_mat.mean())

        return rep_reward.cpu()

    @torch.no_grad()
    def _compute_eff_reward(self, ratio):
        device = self.device

        eff_reward = 1 - torch.tensor(ratio).to(device)
        eff_reward = torch.log(1 + eff_reward)
        eff_reward = torch.clip(eff_reward, 0.0, 0.65)
        return eff_reward.cpu()
