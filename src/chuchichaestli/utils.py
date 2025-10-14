# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various utility functions for chuchichaestli."""

import sys
from functools import partialmethod
import torch.nn as nn
import torch.nn.init as init
from enum import Enum

def partialclass(name: str, cls: type[object], *args, **kwargs):
    """Partial for __init__ class constructors."""
    part_cls = type(
        name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwargs)}
    )
    try:
        part_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass
    return part_cls

class InitMethod(Enum):
    XAVIER = "xavier"
    KAIMING = "kaiming"
    ORTHOGONAL = "orthogonal"
    NORMAL = "normal"
    UNIFORM = "uniform"


def initialize_weights(module, method: InitMethod, gain=0.02):
    """
    Custom weight initialization for Conv and Linear layers.

    Args:
        module: PyTorch module.
        method: One of InitMethod Enum values.
        gain: scaling factor
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        if method == InitMethod.XAVIER:
            init.xavier_uniform_(module.weight.data, gain=gain)
        elif method == InitMethod.KAIMING:
            init.kaiming_normal_(module.weight.data, a=0, mode="fan_in", nonlinearity="relu")
        elif method == InitMethod.ORTHOGONAL:
            init.orthogonal_(module.weight.data, gain=gain)
        elif method == InitMethod.NORMAL:
            init.normal_(module.weight.data, mean=0.0, std=gain)
        elif method == InitMethod.UNIFORM:
            init.uniform_(module.weight.data, -gain, gain)

        if module.bias is not None:
            init.constant_(module.bias.data, 0.0)

