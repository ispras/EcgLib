import os
import yaml
from enum import IntEnum
from urllib.parse import urlparse
from typing import Callable, Dict, List, Union, Optional, Tuple
import dataclasses

import torch
from torch.nn import Module

from .architectures.registred_models import get_builder
from .config.registred_configs import (
    registred_configs,
    config,
    MType,
)
from .config.model_configs import BaseConfig
from .architectures.cnn_tabular import CnnTabular
from .weights.checkpoint import ModelChekpoint

resource_package = __name__

__all__ = [
    "Combination",
    "create_model",
    "get_ecglib_url",
    "get_model",
    "get_config",
    "weights_from_checkpoint",
    "save_checkpoint",
]


class Combination(IntEnum):
    SINGLE = 1
    CNNTAB = 2

    @staticmethod
    def from_string(label: str) -> IntEnum:
        label = label.lower()
        if "single" in label:
            return Combination.SINGLE
        elif "cnntab" in label:
            return Combination.CNNTAB
        else:
            raise ValueError(f"label for combination must be one of [\'single\', \'cnntab\']. Recieved {label}")


def create_model(
    model_name: Union[str, List[str]],
    config: Union[BaseConfig, List[BaseConfig]] = None,
    combine: Union[Combination, str] = Combination.SINGLE,
    pretrained: bool = False,
    pretrained_path: str = "ecglib",
    pathology: Union[str, List[str]] = "AFIB",
    leads_count: int = 12,
    num_classes: int = 1,
) -> Module:
    weights = None
    configs = [config] if isinstance(config, BaseConfig) else config
    model_name = [model_name] if isinstance(model_name, str) else model_name
    combine = Combination.from_string(combine) if isinstance(combine, str) else combine

    if pretrained:  # Currently working only for Combination.SINGLE
        assert (
            combine is Combination.SINGLE
        ), "pretrained is currently working only for Combination.SINGLE"

        assert (
            num_classes == 1
        ), "pretrained is currently working only for binary classification"
        
        if pretrained_path == "ecglib":
            pretrained_path = get_ecglib_url(
                model_name=model_name[0], pathology=pathology, leads_count=leads_count
            )

        weights = weights_from_checkpoint(
            pretrained_path,
            meta_info={
                "pathology": pathology,
                "leads_count": str(leads_count),
                "model_name": model_name[0],
            },
        )

    if combine is Combination.SINGLE:
        assert (
            len(model_name) == 1
        ), "For Combination.SINGLE case `model_name` must contain only one model name"

        if configs:
            configs = configs[0]

        return get_model(name=model_name[0], config=configs, weights=weights)
    elif combine is Combination.CNNTAB:
        assert (
            len(model_name) == 2
        ), f"For Combination.CNNTAB case `model_name` must contain 2 model names (N models in future releases)"

        assert (
            model_name[0] != 'tabular' and model_name[1] == 'tabular'
        ), f"Combination.CNNTAB suggest using a cnn-like architecture as a cnn_backbone part and using TabularNet class as tabular model. Recieved: {model_name[0]}; {model_name[1]}"

        cnn_conf, tab_conf = (
            (None, None) if configs == None else (configs[0], configs[1])
        )

        cnn, cnn_out = get_model(name=model_name[0], config=cnn_conf).get_cnn()
        tab = get_model(name=model_name[1], config=tab_conf)
        tab_out = tab.out_size

        model = CnnTabular(
            cnn_backbone=cnn,
            cnn_out_features=cnn_out,
            tabular_model=tab,
            tabular_out_features=tab_out,
            classes=num_classes,
            head_ftrs=[512],
            head_drop_prob=0.2,
        )

        return model
    else:
        raise ValueError


def get_ecglib_url(model_name: str, pathology: str, leads_count: int):
    try:
        dirname = os.path.dirname(__file__)
        weights_path = os.path.join(dirname, "weights/model_weights_paths.yaml")
        with open(weights_path, "r") as file:
            url = yaml.safe_load(file)[f"{leads_count}_leads"][pathology][model_name]
        return url
    except KeyError as e:
        raise KeyError(f"Key {str(e)} doesn't exist. Check 'ecglib/model/weights/model_weights_paths.yaml\' to see all option for pretrained models.")


def _get_model_builder(name: str) -> Callable[..., Module]:
    """
    Gets the model name and returns the model builder method.

    param: name (str): The name under which the model is registered.
    param: fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = get_builder(name)
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn


def _get_config_stub(model_type: MType) -> BaseConfig:
    """
    Gets the model name and returns the model builder method.

    param: model_name (str): The name under which the model is registered.
    return: config object (BaseConfig)
    """
    try:
        conf_obj = config(model_type)
    except KeyError:
        raise ValueError(
            f"Unknown model type {model_type}. It must be in {registred_configs()}"
        )
    return conf_obj


def get_model(
    name: str, config: Optional[BaseConfig] = None, weights: dict = None
) -> Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    param: name (str): The name under which the model is registered.
    param: config (BaseConfig): Object which contains parameters for the model builder method.
    param: weights (dict): model weights.
    return: model (nn.Module): The initialized model.
    """
    builder = _get_model_builder(name)
    if not config:
        config = get_config(MType.from_string(name))
    model = builder(**config.dict())
    if weights:
        assert isinstance(weights, dict)
        model.load_state_dict(weights, strict=True)
        return model
    return model


def get_config(
    model_type: MType,
    params_overlay: Optional[Dict] = None,
) -> BaseConfig:
    """
    Returns Config class according the model type

    param: model_type(MType): The type of model.
    param: params_overlay (dict): Replace key-values in BaseConfig object with these (NOTE: Only identical keys for `params_overlay` and `BaseConfig instance` will be replaced).
    rerurn config (BaseConfig): BaseConfig instance
    """
    builder = _get_config_stub(model_type)  # get config object
    if params_overlay:
        s1 = set(params_overlay.keys())
        s2 = set(f.name for f in dataclasses.fields(builder))
        identical_keys = s1 & s2
        params_overlay = dict([(key, params_overlay[key]) for key in identical_keys])
        return builder(**params_overlay)
    return builder()


def _checkpoint_from_local(checkpoint_path: str) -> ModelChekpoint:
    """
    Load model info from local checkpoint file.

    param: checkpoint_path(str): Path to checkpoint file.
    return: Dictionary (Dict) which contains information about model checkpoint.
    """
    model_info = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    return ModelChekpoint(model_info)


def _checkpoint_from_remote(url: str, meta_info: dict) -> ModelChekpoint:
    """
    Load model info using remote link.

    param: url(str): Link to remote checkpoint file.
    return: Dictionary (Dict) which contains information about model checkpoint.
    """
    model_info = torch.hub.load_state_dict_from_url(
        url=url,
        map_location='cpu',
        progress=False,
        check_hash=False,
        file_name=f"{meta_info['leads_count']}_leads_{meta_info['model_name']}_{meta_info['pathology']}_1_1_0.pt",
    )
    return ModelChekpoint(model_info)


def weights_from_checkpoint(
    checkpoint_path: str,
    meta_info: dict,
) -> Tuple:
    """
    Return BaseConfig instance from checkpoint file.

    param: checkpoint_path (str): Path to checkpoint file.

    return: config: Tuple which contains models weights, models configs, and experiment information.
    """

    model_info = None
    if urlparse(checkpoint_path).scheme:
        model_info = _checkpoint_from_remote(url=checkpoint_path, meta_info=meta_info)
    else:
        model_info = _checkpoint_from_local(checkpoint_path=checkpoint_path)

    return model_info


def save_checkpoint(path: str, model: Module, info: Dict, exclude: List = None) -> None:
    """
    Save model weights and experiment info as checpoint file

    """
    info["model"] = model.state_dict
    checkpoint = ModelChekpoint.make_checkpoint(model_info=info, exclude_keys=exclude)
    torch.save(checkpoint, path)
