from typing import List, Optional, Callable

from torch.nn import Module

from .resnet1d import resnet1d18, resnet1d50, resnet1d101
from .densenet1d import densenet121_1d, densenet201_1d
from .tabular import tabular

__all__ = ["register_model", "registred_models", "get_builder", "is_model_registred"]


# extensible model's storage
BUILTIN_MODELS = {
    "densenet1d121": densenet121_1d,
    "densenet1d201": densenet201_1d,
    "resnet1d18": resnet1d18,
    "resnet1d50": resnet1d50,
    "resnet1d101": resnet1d101,
    "tabular": tabular,
}


def register_model(
    name: Optional[str] = None,
) -> Callable[[Callable[..., Module]], Callable[..., Module]]:
    """
    Function decorator which helps to register new models
    """

    def wrapper(fn: Callable[..., Module]) -> Callable[..., Module]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper


def registered_models() -> List:
    """
    Returns a list with the names of registered models.
    """
    return list(BUILTIN_MODELS.keys())


def get_builder(name: str) -> Callable[[Callable[..., Module]], Callable[..., Module]]:
    """
    Returns a model builder callable object.

    param: name (str): Model name.
    return: Callable
    """
    if name not in BUILTIN_MODELS:
        raise ValueError(
            f"An entry is not registered in `BUILTIN_MODELS`. Available models: {registered_models()}."
        )
    return BUILTIN_MODELS[name]


def is_model_registred(name: str) -> bool:
    """
    Checks is model was registered.

    param: name (str): Model name.
    return: Boolean flag.
    """
    return name in BUILTIN_MODELS
