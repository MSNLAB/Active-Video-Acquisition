import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .utils import to_cpu_and_share_memory, to_device


def _inputs_requries_grad(inputs):
    return any(t.requires_grad for t in inputs if isinstance(t, torch.Tensor))


def _module_all_requires_grad(module):
    return all(p.requires_grad for p in module.parameters())


def _module_no_requires_grad(module):
    return all(not p.requires_grad for p in module.parameters())


def _module_any_requires_grad(module):
    return any(p.requires_grad for p in module.parameters())


class ForwardCachedModule(nn.Module):
    def __init__(self, module, use_cache=False):
        super(ForwardCachedModule, self).__init__()
        self.module = module
        self.use_cache = use_cache
        self.forward_cache = []

    def forward(self, *inputs):
        if _inputs_requries_grad(inputs) or _module_any_requires_grad(self.module):
            return self.module(*inputs)

        with torch.no_grad():
            if self.use_cache:
                return self.forward_cache.pop(0)

            self.forward_cache.append(self.module(*inputs))
            return self.forward_cache[-1]


class ForwardCachedModel(nn.Module):
    def __init__(self, model):
        super(ForwardCachedModel, self).__init__()
        self.model = model
        self.forward_cache_modules = []
        self._wrapper_model(self.model, self.forward_cache_modules)

    def _wrapper_model(self, model, forward_cache_modules):
        for name, module in model.named_children():
            if len(list(module.parameters())) == 0:
                continue

            if (
                len(list(module.children())) == 0
                or _module_all_requires_grad(module)
                or _module_no_requires_grad(module)
            ):
                warpper_module = ForwardCachedModule(module)
                model.add_module(name, warpper_module)
                forward_cache_modules.append(warpper_module)
            else:
                self._wrapper_model(module, forward_cache_modules)

    def get_forward_caches(self):
        forward_caches = []
        for module in self.forward_cache_modules:
            forward_caches.append(module.forward_cache)
        return forward_caches

    def clear_forward_caches(self):
        for module in self.forward_cache_modules:
            module.forward_cache.clear()

    def forward(self, *args, forward_caches=None, **kwargs):
        if forward_caches is None:
            return self.no_cache_forward(*args, **kwargs)
        return self.cache_forward(*args, forward_caches=forward_caches, **kwargs)

    def cache_forward(self, *args, forward_caches, **kwargs):
        for module, forward_cache in zip(self.forward_cache_modules, forward_caches):
            module.use_cache = True
            module.forward_cache = forward_cache
        return self.model(*args, **kwargs)

    def no_cache_forward(self, *args, **kwargs):
        for module in self.forward_cache_modules:
            module.use_cache = False
            module.forward_cache.clear()
        return self.model(*args, **kwargs)


class ForwardCachedDataset(Dataset):
    def __init__(self, dataset, select_ids=None):
        self.dataset = to_cpu_and_share_memory([data for data in dataset])
        self.dataset_collate_fn = dataset.collate_fn
        self.forward_caches = [None for _ in range(len(dataset))]

        if select_ids is None:
            select_ids = list(range(len(dataset)))
        self.select_ids = select_ids

    def __len__(self):
        return len(self.select_ids)

    def __getitem__(self, index):
        index = self.select_ids[index]
        return index, self.dataset[index], self.forward_caches[index]

    @torch.no_grad()
    def update_caches(self, index_batch, forward_caches_batch):
        for offset in range(len(index_batch)):
            index = index_batch[offset]
            forward_caches = []

            for forward_cache_batch in forward_caches_batch:
                _forward_cache = []
                for forward_cache in forward_cache_batch:
                    if isinstance(forward_cache, torch.Tensor):
                        _forward_cache.append(forward_cache[offset])
                    elif isinstance(forward_cache, dict):
                        _forward_cache.append({k: v[offset] for k, v in forward_cache.items()})
                forward_caches.append(_forward_cache)

            forward_caches = to_cpu_and_share_memory(forward_caches)
            self.forward_caches[index] = forward_caches

    @torch.no_grad()
    def collate_fn(self, batches, device=None, non_blocking=True):
        ids, data, caches = tuple(zip(*[batch for batch in batches]))

        if device is None:
            device = torch.cuda.current_device()

        ids = to_device(torch.tensor(ids), device, non_blocking)
        data = to_device(self.dataset_collate_fn(data), device, non_blocking)

        if any(cache is None for cache in caches):
            return ids, data, None

        forward_caches_batch = to_device(list(zip(*caches)), device, non_blocking)
        for offset in range(len(forward_caches_batch)):
            forward_caches = []
            for forward_cache in list(zip(*forward_caches_batch[offset])):
                if isinstance(forward_cache[0], torch.Tensor):
                    forward_caches.append(torch.stack(forward_cache))
                elif isinstance(forward_cache[0], dict):
                    forward_caches.append(
                        {k: torch.stack([cache[k] for cache in forward_cache]) for k in forward_cache[0].keys()}
                    )
            forward_caches_batch[offset] = forward_caches
        caches = forward_caches_batch

        return ids, data, caches
