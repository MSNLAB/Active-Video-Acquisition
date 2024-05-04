import gc
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


@torch.no_grad()
def to_device(tensors, device="cpu", non_blocking=False):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device, non_blocking=non_blocking)
    elif isinstance(tensors, tuple):
        return [to_device(t, device, non_blocking) for t in tensors]
    elif isinstance(tensors, list):
        return [to_device(t, device, non_blocking) for t in tensors]
    elif isinstance(tensors, dict):
        return {k: to_device(v, device, non_blocking) for k, v in tensors.items()}
    return tensors


@torch.no_grad()
def to_cpu_and_share_memory(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.cpu()  # .share_memory_()
    elif isinstance(tensors, tuple):
        return [to_cpu_and_share_memory(t) for t in tensors]
    elif isinstance(tensors, list):
        return [to_cpu_and_share_memory(t) for t in tensors]
    elif isinstance(tensors, dict):
        return {k: to_cpu_and_share_memory(v) for k, v in tensors.items()}
    return tensors


def same_seeds(seed=42069):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def find_leaf_modules(module):
    leaf_modules = []
    for child in module.children():
        if len(list(child.children())) == 0:
            leaf_modules.append(child)
        else:
            leaf_modules.extend(find_leaf_modules(child))
    return leaf_modules


def cache_empty(func):
    def wrapper_func(*args, **kwargs):
        res = func(*args, **kwargs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res

    return wrapper_func


def set_module_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad
        if requires_grad is False and param.grad is not None:
            param.grad = None


def weights_init_kaiming(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
        nn.init.constant_(module.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if module.affine is not None:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def weights_init_classifier(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


@torch.no_grad()
def grad_norm(parameter_groups):
    norms = []
    for parameter_group in parameter_groups:
        norm = 0
        for p in parameter_group["params"]:
            norm += p.grad.data.norm(2) ** 2
        norm = norm ** (1.0 / 2)
        norms.append(norm)
    return sum(norms)


def sync_dist_model(model: nn.Module):
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        for _, p in model.state_dict(keep_vars=True).items():
            dist.broadcast(p, src=0)


def sync_cuda_rng_state():
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        rng_state = torch.get_rng_state().cuda()
        cuda_rng_state = torch.cuda.get_rng_state().cuda()
        dist.broadcast(rng_state, src=0)
        dist.broadcast(cuda_rng_state, src=0)
        torch.set_rng_state(rng_state.byte().cpu())
        torch.cuda.set_rng_state(cuda_rng_state.byte().cpu())
