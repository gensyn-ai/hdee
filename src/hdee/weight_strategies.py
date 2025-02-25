import torch
import logging
import torch.distributed as dist

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)


class UniformStrategy:
    def __init__(self):
        self._weight = 1.0 / (dist.get_world_size())

    def __call__(self):
        return self._weight


class UniformStrategyBuilder:
    def __init__(self):
        pass

    def build(self, *args, **kwargs):
        return UniformStrategy()
