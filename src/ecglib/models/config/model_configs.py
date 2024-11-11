from ..architectures.resnet1d import BottleneckBlock1d
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn

__all__ = [
    "BaseConfig",
    "ResNetConfig",
    "TabularNetConfig",
    "DenseNetConfig",
    "CNN1dConfig",
    "SSSDConfig"
]


@dataclass
class BaseConfig:
    """
    Base congiguration class
    """

    checkpoint_url: Optional[Union[str, Tuple[str, str]]] = None
    checkpoint_file: Optional[str] = None
    configs_hub: Optional[str] = None

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_url or self.checkpoint_file or self.configs_hub

    def dict(self) -> Dict:
        return {f: getattr(self, f) for f in self.__annotations__}

    def load_checkpoint(self) -> Any:
        assert self.has_checkpoint()
        if self.checkpoint_file:
            return torch.load(self.checkpoint_file)
        elif self.checkpoint_url:
            raise NotImplementedError
        else:
            raise FileNotFoundError

    def load_from_hub(self):
        # Must be implemented in child classes to load checkpoints from `configs_hub`
        raise NotImplementedError


@dataclass(repr=True, eq=True)
class ResNetConfig(BaseConfig):
    """
    Default parameters correspond Resnet1d50 model
    """

    block_type: nn.Module = field(default_factory=lambda: BottleneckBlock1d)
    layers: list = field(default_factory=lambda: [3, 4, 6, 3])
    input_channels: int = 12
    base_filters: int = 64
    kernel_size: int = 3
    stride: int = 2
    num_classes: int = 1
    dropout_prob: float = 0.0
    fix_feature_dim: bool = True
    kernel_size_stem: Optional[int] = None
    stride_stem: int = 2
    pooling_stem: bool = True
    concat_pooling: bool = True
    hidden_layers_head: Optional[list] = None
    dropout_prob_head: float = 0.5
    act_head: str = "relu"
    bn_head: bool = True
    bn_final_head: bool = False


@dataclass(repr=True, eq=True)
class TabularNetConfig(BaseConfig):
    """
    Default parameters correspond TabularNet model
    """

    inp_features: int = 5
    lin_ftrs: list = field(
        default_factory=lambda: [10, 10, 10, 10, 10, 8],
    )
    drop: Union[int, str] = field(default_factory=lambda: [0.5])
    act_fn: str = field(default_factory=lambda: "relu")
    bn_last: bool = field(default_factory=lambda: True)
    act_last: bool = field(default_factory=lambda: True)
    drop_last: bool = field(default_factory=lambda: True)


@dataclass(repr=True, eq=True)
class DenseNetConfig(BaseConfig):
    """
    Default parameters correspond DenseNet121_1d model
    """

    inp_features: int = 5000
    growth_rate: int = 32
    block_config: tuple = (6, 12, 24, 16)
    num_init_features: int = 64
    bottleneck_size: int = 4
    kernel_size: int = 3
    input_channels: int = 12
    num_classes: int = 1
    reinit: bool = True


@dataclass(repr=True, eq=True)
class CNN1dConfig(BaseConfig):
    inp_channels: int = 12
    inp_features: int = 1
    cnn_ftrs: list = field(default_factory=lambda: [64, 32, 16])


@dataclass(repr=True, eq=True)
class SSSDConfig(BaseConfig):
    """
    Default parameters correspond SSSD_ECG model
    """
    in_channels: int = 8
    res_channels: int = 256
    skip_channels: int = 256
    out_channels: int = 8
    num_res_layers: int = 36
    diffusion_step_embed_dim_in: int = 128
    diffusion_step_embed_dim_mid: int = 512
    diffusion_step_embed_dim_out: int = 512
    s4_lmax: int = 1000
    s4_d_state: int = 64
    s4_dropout: float = 0.0
    s4_bidirectional: bool = True
    s4_layernorm: bool = True
    label_embed_dim: int = 128
    label_embed_classes: int = 40
    gender_embed_classes: int = 0
    gender_embed_dim: int = 128
    new_label_embed: bool = True
