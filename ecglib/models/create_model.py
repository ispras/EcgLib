import os
from collections import OrderedDict
from dataclasses import asdict
from operator import mod

import hydra
import torch
from omegaconf import DictConfig
import yaml

from .architectures.densenet1d import densenet121_1d, densenet201_1d
from .config.model_configs import DenseNetConfig

resource_package = __name__

pathologies = ["AFIB", "STACH", "SBRAD", "RBBB", "LBBB", "PVC", "1AVB"]

arch_map = {
    "densenet1d121": densenet121_1d,
    "densenet1d201": densenet201_1d,
}


def get_config(model_name: str, config: dict = None):
    if config:
        model_name = config.model_name
        return hydra.utils.instantiate(config.config)

    if "densenet" in model_name:
        return DenseNetConfig()
    else:
        raise Exception("Unknown model type")


def get_model(
    model_name: str,
    leads_num: int,
    model_cfg: DictConfig,
    num_classes: int,
) -> torch.nn.Module:
    if model_cfg:
        model_name = model_cfg.model_name

    assert (
        model_name in arch_map
    ), "Model name must be one of ['densenet1d121', 'densenet1d201']"

    model_config = get_config(model_name=model_name, config=model_cfg)
    model_config.input_channels = leads_num
    model_config.num_classes = num_classes

    # Note: Overloads config params according to input args values
    if isinstance(model_config, DenseNetConfig):
        return arch_map[model_name](**asdict(model_config))


def create_model(
    model_name: str,
    pathology: str,
    model_cfg: dict = None,
    pretrained: bool = False,
    leads_num: int = 12,
):
    model = get_model(
        model_name=model_name,
        leads_num=leads_num,
        model_cfg=model_cfg,
        num_classes=1,
        # meta=meta,
    )

    if pretrained:
        if pathology not in pathologies:
            raise KeyError(
                "pathology must be one of ['AFIB', 'STACH', 'SBRAD', 'RBBB', 'LBBB', 'PVC', '1AVB']"
            )

        dirname = os.path.dirname(__file__)
        weights_path = os.path.join(dirname, 'weights/model_weights_paths.yaml')
        with open(weights_path, 'r') as file:
            weights_config = yaml.safe_load(file)
        if model_name not in weights_config[f"{leads_num}_leads"][pathology]:
            raise KeyError(
                "the weights are currently available for the following architectures ['densenet1d121']"
            )
        weights_path = weights_config[f"{leads_num}_leads"][pathology][model_name]

        model_info = torch.hub.load_state_dict_from_url(weights_path, progress=True, check_hash=True, file_name=f"{leads_num}_leads_{model_name}_{pathology}.pt")

        print('{} model trained on {}-lead {} second ECG records with {} frequency to detect {}. {} normalization was applied to all the records during preprocessing.'.format(
            model_name,
            leads_num,
            model_info['config_file']['ecg_record_params']['observed_ecg_length'],
            model_info['config_file']['ecg_record_params']['resampled_frequency'],
            pathology,
            model_info['config_file']['ecg_record_params']['normalization'],
        ))


        model_state_dict = model_info['model']
        model.load_state_dict(model_state_dict)
    return model
