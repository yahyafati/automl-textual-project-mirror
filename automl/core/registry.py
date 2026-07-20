from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Dict, Type
from typing import Iterable

from automl.core.approaches.base_approach import Approach
from automl.core.types import ApproachName
from automl.logger import get_logger

_APPROACH_REGISTRY: Dict[ApproachName, Type[Approach]] = {}

logger = get_logger()


def _iter_submodules(package: ModuleType) -> Iterable[str]:
    package_name = package.__name__
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        yield f"{package_name}.{info.name}"


def register_all_approaches() -> Dict[str, Type[Approach]]:
    from . import approaches

    logger.info("Initializing approach discovery...")

    for module_name in _iter_submodules(approaches):
        logger.debug(f"Importing submodule to trigger registration: {module_name}")
        importlib.import_module(module_name)

    logger.info(
        f"Finished discovery. {len(_APPROACH_REGISTRY)} approach(es) registered."
    )
    return _APPROACH_REGISTRY


def register_approach(name: ApproachName):
    """
    Class decorator to register a new approach under a given name.

    Example:
        @register_approach("tfidf-ffnn")
        class TfidfApproach:
            ...
    """

    def decorator(cls: Type[Approach]) -> Type[Approach]:
        if name in _APPROACH_REGISTRY:
            logger.error(
                f"Registration conflict: Approach '{name}' is already registered to {_APPROACH_REGISTRY[name].__name__}."
            )
            raise ValueError(f"Approach '{name}' already registered")

        _APPROACH_REGISTRY[name] = cls
        setattr(cls, "name", name)
        # logger.debug(f"Registered approach: '{name}' -> {cls.__name__}")

        return cls

    return decorator


def get_approach(name: ApproachName) -> Type[Approach]:
    try:
        return _APPROACH_REGISTRY[name]
    except KeyError:
        logger.error(f"Failed to fetch approach '{name}'. It has not been registered.")
        raise ValueError(
            f"Unknown approach '{name}'. Registered: {sorted(_APPROACH_REGISTRY)}. "
            f"Did you forget to run 'automl.registry.register_all_approaches()'"
        ) from None


def list_approaches() -> Dict[ApproachName, Type[Approach]]:
    """
    Returns a copy of the current registry.
    Helpful for debugging or introspection.
    """
    return dict(_APPROACH_REGISTRY)
