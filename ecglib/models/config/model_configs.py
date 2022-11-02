import imp
from dataclasses import dataclass
from typing import Optional, Type

import torch.nn as nn


@dataclass(repr=True, eq=True)
class DenseNetConfig:
    """
    Default parameters correspond DenseNet121_1d model
    """

    growth_rate: int = 32
    block_config: tuple = (6, 12, 24, 16)
    num_init_features: int = 64
    bottleneck_size: int = 4
    kernel_size: int = 3
    input_channels: int = 12
    num_classes: int = 1
    reinit: bool = True