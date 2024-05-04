import copy
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.detection import MeanAveragePrecision

from models import *
from utils import *

logger = get_logger(__file__)


class FRCNNTrainer:
    def __init__(
        self,
        model,
        tr_dataset,
        val_dataset,
        parameter_groups=None,
        amp_enable=True,
        forward_cache=False,
        verbose=False,
    ):
        self.reset_model(model, parameter_groups)
        self.tr_dataset = ForwardCachedDataset(tr_dataset)
        self.val_dataset = ForwardCachedDataset(val_dataset)
        self.prefetcher = DataloaderPrefetcher()
        self.amp_enable = amp_enable
        self.forward_cache = forward_cache
        self.metric_calculator = MeanAveragePrecision()
        self.verbose = verbose
        if self.forward_cache:
            self._update_forward_caches()

        self.rng_state = torch.cuda.get_rng_state().cuda()
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.rng_state, src=0)
        self.rng_state = self.rng_state.byte().cpu()

    def reset_model(self, model, param_groups=None):
        if param_groups is None:
            param_groups = [{"params": [param for param in model.parameters()]}]

        self.model = ForwardCachedModel(model)
        self.optimizer = Adam(param_groups)
        self.scaler = GradScaler()

        self.initial_params = {n: p for n, p in self.model.state_dict().items()}

    def reset_params(self, param_groups=None):
        if param_groups is None:
            param_groups = self.initial_params

        self.model.load_state_dict(param_groups)
        self.optimizer.state = defaultdict(dict)
        self.scaler = GradScaler()

    @cache_empty
    def _update_forward_caches(self, batch_size=8, device=None):
        if device is None:
            device = torch.cuda.current_device()

        model = copy.deepcopy(self.model).to(device)
        dataloader = DataLoader(
            dataset=self.tr_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
        )

        model.train()
        for ids, data, caches in self.prefetcher(dataloader, self.tr_dataset.collate_fn):
            model.no_cache_forward(*data)
            if caches is None:
                self.tr_dataset.update_caches(ids, model.get_forward_caches())
            model.clear_forward_caches()

        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: x,
        )

        model.train()
        for ids, data, caches in self.prefetcher(dataloader, self.val_dataset.collate_fn):
            model.no_cache_forward(*data)
            if caches is None:
                self.val_dataset.update_caches(ids, model.get_forward_caches())
            model.clear_forward_caches()

        model.cpu()

    def step(self, epoch_cnt=10, batch_size=8, early_stop=None, select_ids=None, pin_memory=False, device=None):
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.rng_state)

        if device is None:
            device = torch.cuda.current_device()

        if select_ids is None:
            select_ids = list(range(len(self.tr_dataset)))
        self.tr_dataset.select_ids = select_ids

        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
            model = nn.parallel.DistributedDataParallel(self.model.to(rank), device_ids=[rank])
            sampler = DistributedSampler(self.tr_dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(
                dataset=self.tr_dataset,
                batch_size=batch_size // world_size,
                collate_fn=lambda x: x,
                pin_memory=pin_memory,
                sampler=sampler,
            )
        else:
            model = self.model.to(device)
            sampler = RandomSampler(self.tr_dataset)
            dataloader = DataLoader(
                dataset=self.tr_dataset,
                batch_size=batch_size,
                collate_fn=lambda x: x,
                pin_memory=pin_memory,
                sampler=sampler,
            )

        model.train()
        best_loss, best_epoch = None, None
        for epoch in range(epoch_cnt):
            metric_state = {
                "loss": 0.0,
                "loss/classifier": 0.0,
                "loss/box_reg": 0.0,
                "loss/objectness": 0.0,
                "loss/rpn_box_reg": 0.0,
                "grad_norm": 0.0,
                "time": 0.0,
                "time/forward": 0.0,
                "time/backward": 0.0,
                "time/dataload": 0.0,
            }

            with tc.collect_time("time"):
                data_iter = self.prefetcher(dataloader, self.tr_dataset.collate_fn)
                while True:
                    try:
                        with tc.collect_time("time/dataload"):
                            _, data, caches = next(data_iter)

                        with tc.collect_time("time/forward"):
                            with autocast(enabled=self.amp_enable):
                                losses = model(*data, forward_caches=caches)
                            loss = sum(losses.values())

                        with tc.collect_time("time/backward"):
                            self.optimizer.zero_grad()
                            if self.amp_enable:
                                self.scaler.scale(loss).backward()
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                loss.backward()
                                self.optimizer.step()

                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                        metric_state["loss"] += loss.item() / len(dataloader)

                        grad_nrom = grad_norm(self.optimizer.param_groups)
                        metric_state["grad_norm"] += grad_nrom / len(dataloader)

                        for metric, value in losses.items():
                            metric = metric.replace("loss_", "loss/")
                            if dist.is_available() and dist.is_initialized():
                                dist.all_reduce(value, op=dist.ReduceOp.AVG)
                            metric_state[metric] += value.item() / len(dataloader)

                        metric_state["time/dataload"] += tc.times["time/dataload"]
                        metric_state["time/forward"] += tc.times["time/forward"]
                        metric_state["time/backward"] += tc.times["time/backward"]

                    except StopIteration:
                        break

            metric_state["time"] += tc.times["time"]

            if self.verbose:
                logger.info(f"[{epoch + 1}/{epoch_cnt}]", extra={"metrics": metric_state})

            if early_stop:
                if best_loss is None or metric_state["loss"] < best_loss:
                    best_loss = metric_state["loss"]
                    best_epoch = epoch

                if epoch - best_epoch > early_stop:
                    break

        torch.cuda.set_rng_state(rng_state)

    def eval(self, select_ids=None, batch_size=8, pin_memory=False, device=None):
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.rng_state)

        if device is None:
            device = torch.cuda.current_device()

        if select_ids is None:
            select_ids = list(range(len(self.val_dataset)))
        self.val_dataset.select_ids = select_ids

        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
            model = nn.parallel.DistributedDataParallel(self.model.to(rank), device_ids=[rank])
            sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=batch_size // world_size,
                collate_fn=lambda x: x,
                pin_memory=pin_memory,
                sampler=sampler,
            )
        else:
            model = self.model.to(device)
            sampler = RandomSampler(self.val_dataset)
            dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=batch_size,
                collate_fn=lambda x: x,
                pin_memory=pin_memory,
                sampler=sampler,
            )

        model.eval()
        metric_state = {
            "map": 0.0,
            "map/50": 0.0,
            "map/75": 0.0,
            "map/small": 0.0,
            "map/medium": 0.0,
            "map/large": 0.0,
            "mar/1": 0.0,
            "mar/10": 0.0,
            "mar/100": 0.0,
            "mar/small": 0.0,
            "mar/medium": 0.0,
            "mar/large": 0.0,
            "time": 0.0,
            "time/forward": 0.0,
            "time/dataload": 0.0,
            "time/compute_map": 0.0,
        }

        with tc.collect_time("time"):
            data_iter = self.prefetcher(dataloader, self.val_dataset.collate_fn)
            while True:
                try:
                    with tc.collect_time("time/dataload"):
                        _, data, caches = next(data_iter)

                    with tc.collect_time("time/forward"):
                        with autocast(enabled=self.amp_enable):
                            images, targets = data
                            detections = model(images, targets, forward_caches=caches)

                        self.metric_calculator.update(detections, targets)

                except StopIteration:
                    break

            with tc.collect_time("time/compute_map"):
                mAP = self.metric_calculator.compute()
                self.metric_calculator.reset()

        metric_state["time"] += tc.times["time"]
        metric_state["time/forward"] += tc.times["time/forward"]
        metric_state["time/dataload"] += tc.times["time/dataload"]
        metric_state["time/compute_map"] += tc.times["time/compute_map"]
        metric_state["map"] += mAP["map"]
        metric_state["map/50"] += mAP["map_50"]
        metric_state["map/75"] += mAP["map_75"]
        metric_state["map/small"] += mAP["map_small"]
        metric_state["map/medium"] += mAP["map_medium"]
        metric_state["map/large"] += mAP["map_large"]
        metric_state["mar/1"] += mAP["mar_1"]
        metric_state["mar/10"] += mAP["mar_10"]
        metric_state["mar/100"] += mAP["mar_100"]
        metric_state["mar/small"] += mAP["mar_small"]
        metric_state["mar/medium"] += mAP["mar_medium"]
        metric_state["mar/large"] += mAP["mar_large"]

        if self.verbose:
            logger.info(
                f"IoU metric: bbox; \n"
                f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mAP['map']:.3f} \n"
                f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {mAP['map_50']:.3f} \n"
                f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {mAP['map_75']:.3f} \n"
                f"Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {mAP['map_small']:.3f} \n"
                f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {mAP['map_medium']:.3f} \n"
                f"Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {mAP['map_large']:.3f} \n"
                f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {mAP['mar_1']:.3f} \n"
                f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {mAP['mar_10']:.3f} \n"
                f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mAP['mar_100']:.3f} \n"
                f"Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {mAP['mar_small']:.3f} \n"
                f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {mAP['mar_medium']:.3f} \n"
                f"Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {mAP['mar_large']:.3f} \n"
            )

        torch.cuda.set_rng_state(rng_state)

        return metric_state
