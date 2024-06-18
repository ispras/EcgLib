from .model_configs import (
    BaseConfig,
    ResNetConfig,
    DenseNetConfig,
    TabularNetConfig,
    SSSDConfig,
)

from ..architectures.model_types import MType

from typing import List, Any

__all__ = ["register_config", "registred_configs", "config", "is_conf_registred"]

# extensible config's storage
BUILTIN_CONFIGS = {
    MType.RESNET: ResNetConfig,
    MType.DENSENET: DenseNetConfig,
    MType.TABULAR: TabularNetConfig,
    MType.SSSD: SSSDConfig
}


def register_config(
    model_type: MType,
) -> Any:
    """
    Function decorator which helps to register new configs
    """

    def wrapper(config_obj: Any) -> Any:
        # key = model_type if model_type is not None else config_obj.__name__
        if model_type in BUILTIN_CONFIGS:
            raise ValueError(
                f"An entry is already registered under the name '{model_type}'."
            )
        BUILTIN_CONFIGS[model_type] = config_obj
        return config_obj

    return wrapper


def registred_configs() -> List[str]:
    """
    Returns a list with the names of registered configs.
    """
    return list(BUILTIN_CONFIGS.keys())


def config(model_type: MType) -> BaseConfig:
    """
    Returns config object class.

    param: name (str): Model name.
    return: config (BaseConfig) object
    """

    if model_type not in BUILTIN_CONFIGS:
        raise ValueError(
            f"An entry is not registered in `BUILTIN_CONFIGS`. Available configs: {registred_configs()}."
        )
    return BUILTIN_CONFIGS[model_type]


def is_conf_registred(model_type: MType) -> bool:
    """
    Checks is model was registered.

    param: name (str): Model name.
    return: Boolean flag.
    """
    return model_type in BUILTIN_CONFIGS
