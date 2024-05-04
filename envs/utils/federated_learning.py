from typing import List

import torch
import torch.nn as nn


@torch.no_grad()
def aggregate_grads(models, weights=None):
    if weights is None:
        weights = [1 for _ in models]

    merged_grads = {}
    for state in zip(*[m.named_parameters() for m in models]):
        name = state[0][0]
        grads = sum([w * s[1].grad for w, s in zip(weights, state)]) / sum(weights)
        merged_grads[name] = grads
    return merged_grads


@torch.no_grad()
def aggregate_params(models, weights=None):
    if weights is None:
        weights = [1 for _ in models]

    merged_params = {}
    for state in zip(*[m.state_dict().items() for m in models]):
        name = state[0][0]
        params = sum([w * s[1] for w, s in zip(weights, state)]) / sum(weights)
        merged_params[name] = params
    return merged_params
