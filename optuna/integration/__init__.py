from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any
from typing import TYPE_CHECKING

from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


_import_structure = {
    "botorch": ["BoTorchSampler"],
    "catboost": ["CatBoostPruningCallback"],
    "cma": ["PyCmaSampler"],
    "dask": ["DaskStorage"],
    "mlflow": ["MLflowCallback"],
    "wandb": ["WeightsAndBiasesCallback"],
    "keras": ["KerasPruningCallback"],
    "lightgbm": ["LightGBMPruningCallback", "LightGBMTuner", "LightGBMTunerCV"],
    "pytorch_distributed": ["TorchDistributedTrial"],
    "pytorch_ignite": ["PyTorchIgnitePruningHandler"],
    "pytorch_lightning": ["PyTorchLightningPruningCallback"],
    "sklearn": ["OptunaSearchCV"],
    "shap": ["ShapleyImportanceEvaluator"],
    "skorch": ["SkorchPruningCallback"],
    "tensorboard": ["TensorBoardCallback"],
    "tensorflow": ["TensorFlowPruningHook"],
    "tfkeras": ["TFKerasPruningCallback"],
    "xgboost": ["XGBoostPruningCallback"],
    "fastaiv2": ["FastAIV2PruningCallback", "FastAIPruningCallback"],
}


__all__ = [
    "BoTorchSampler",
    "CatBoostPruningCallback",
    "PyCmaSampler",
    "DaskStorage",
    "MLflowCallback",
    "WeightsAndBiasesCallback",
    "KerasPruningCallback",
    "LightGBMPruningCallback",
    "LightGBMTuner",
    "LightGBMTunerCV",
    "TorchDistributedTrial",
    "PyTorchIgnitePruningHandler",
    "PyTorchLightningPruningCallback",
    "OptunaSearchCV",
    "ShapleyImportanceEvaluator",
    "SkorchPruningCallback",
    "TensorBoardCallback",
    "TensorFlowPruningHook",
    "TFKerasPruningCallback",
    "XGBoostPruningCallback",
    "FastAIV2PruningCallback",
    "FastAIPruningCallback",
]


if TYPE_CHECKING:
    from optuna.integration.botorch import BoTorchSampler
    from optuna.integration.catboost import CatBoostPruningCallback
    from optuna.integration.cma import PyCmaSampler
    from optuna.integration.dask import DaskStorage
    from optuna.integration.fastaiv2 import FastAIPruningCallback
    from optuna.integration.fastaiv2 import FastAIV2PruningCallback
    from optuna.integration.keras import KerasPruningCallback
    from optuna.integration.lightgbm import LightGBMPruningCallback
    from optuna.integration.lightgbm import LightGBMTuner
    from optuna.integration.lightgbm import LightGBMTunerCV
    from optuna.integration.mlflow import MLflowCallback
    from optuna.integration.pytorch_distributed import TorchDistributedTrial
    from optuna.integration.pytorch_ignite import PyTorchIgnitePruningHandler
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
    from optuna.integration.shap import ShapleyImportanceEvaluator
    from optuna.integration.sklearn import OptunaSearchCV
    from optuna.integration.skorch import SkorchPruningCallback
    from optuna.integration.tensorboard import TensorBoardCallback
    from optuna.integration.tensorflow import TensorFlowPruningHook
    from optuna.integration.tfkeras import TFKerasPruningCallback
    from optuna.integration.wandb import WeightsAndBiasesCallback
    from optuna.integration.xgboost import XGBoostPruningCallback
else:

    class _IntegrationModule(ModuleType):

        __all__ = __all__
        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        _modules = set(_import_structure.keys())
        _class_to_module = {}
        for key, values in _import_structure.items():
            for value in values:
                _class_to_module[value] = key

        def __getattr__(self, name: str) -> Any:
            if name in self._modules:
                value = self._get_module(name)
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            else:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")

            setattr(self, name, value)
            return value


    sys.modules[__name__] = _IntegrationModule(__name__)
