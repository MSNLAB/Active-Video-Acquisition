import copy
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from models import *
from utils import *

logger = get_logger(__file__)


def top_k_accuracy(output, target, k):
    with torch.no_grad():
        _, pred = output.topk(k, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.float().sum()


class ResNetTrainer:
    def __init__(
        self,
        model,
        tr_dataset,
        val_dataset,
        parameter_groups=None,
        amp_enable=True,
        forward_cache=True,
        verbose=False,
    ):
        self.reset_model(model, parameter_groups)
        self.tr_dataset = ForwardCachedDataset(tr_dataset)
        self.val_dataset = ForwardCachedDataset(val_dataset)
        self.prefetcher = DataloaderPrefetcher()
        self.amp_enable = amp_enable
        self.forward_cache = forward_cache
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
    def _update_forward_caches(self, batch_size=128, device=None):
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

    def step(self, epoch_cnt=10, batch_size=128, early_stop=None, select_ids=None, pin_memory=False, device=None):
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
                                loss = model(*data, forward_caches=caches)

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

    def eval(self, select_ids=None, batch_size=128, pin_memory=False, device=None):
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
            "acc/top_1": 0.0,
            "acc/top_3": 0.0,
            "acc/top_5": 0.0,
            "time": 0.0,
            "time/forward": 0.0,
            "time/dataload": 0.0,
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
                            preds = model(images, targets, forward_caches=caches)

                        top_1_acc = top_k_accuracy(preds, targets, 1)
                        top_3_acc = top_k_accuracy(preds, targets, 3)
                        top_5_acc = top_k_accuracy(preds, targets, 5)

                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(top_1_acc, op=dist.ReduceOp.AVG)
                            dist.all_reduce(top_3_acc, op=dist.ReduceOp.AVG)
                            dist.all_reduce(top_5_acc, op=dist.ReduceOp.AVG)

                        metric_state["acc/top_1"] += top_1_acc.item() / len(self.val_dataset)
                        metric_state["acc/top_3"] += top_3_acc.item() / len(self.val_dataset)
                        metric_state["acc/top_5"] += top_5_acc.item() / len(self.val_dataset)

                except StopIteration:
                    break

        metric_state["time"] += tc.times["time"]
        metric_state["time/forward"] += tc.times["time/forward"]
        metric_state["time/dataload"] += tc.times["time/dataload"]

        if self.verbose:
            logger.info(
                "\nAccuracy \n"
                f"Top-1 Accuracy = {metric_state['acc/top_1']:.3f} \n"
                f"Top-3 Accuracy = {metric_state['acc/top_3']:.3f} \n"
                f"Top-5 Accuracy = {metric_state['acc/top_5']:.3f} \n"
            )

        torch.cuda.set_rng_state(rng_state)

        return metric_state
