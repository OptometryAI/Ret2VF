from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    'seed': 42,
    'reproducibility': {
        'deterministic': True,
        'warn_only': True,
    },
    'device': 'cuda',
    'task': {
        'task_type': 'c',
        'predict_classes': True,
        'use_weight': False,
    },
    'loss': {
        'main_loss': 'ordinal_bce',
        'main_reduction': 'mean',
        'pm_focal_alpha': 0.7,
        'pm_focal_gamma': 2.0,
        'pm_loss_weight': 1.0,
        'md_loss_weight': 1.0,
    },
    'optimizer': {
        'name': 'adam',
        'lr': 1e-4,
        'weight_decay': 0.0,
    },
    'scheduler': {
        'name': 'none',
        'step_on': 'val_main_loss',
    },
    'train': {
        'epochs': 100,
        'use_test': True,
        'checkpoint': 0,
        'max_grad_norm': None,
    },
    'metrics': {
        'ordinal_threshold': 0.5,
    },
    'logging': {
        'use_tensorboard': True,
    },
    'selection': {
        'monitor': 'rmse',
    },
    'save': {
        'save_root': 'save_result',
        'task_name': 'default_task',
    },
}


REQUIRED_TOP_LEVEL_KEYS = ['model', 'data']


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _validate_config(cfg: Dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing required top-level config section: '{key}'")

    if cfg['task']['task_type'] not in {'r', 'c'}:
        raise ValueError("task.task_type must be 'r' or 'c'.")

    if cfg['optimizer']['lr'] <= 0:
        raise ValueError('optimizer.lr must be positive.')

    if cfg['train']['epochs'] <= 0:
        raise ValueError('train.epochs must be positive.')

    if cfg['scheduler']['name'] not in {'none', 'reduce_on_plateau', 'step', 'multistep', 'cosine'}:
        raise ValueError("scheduler.name must be one of {'none', 'reduce_on_plateau', 'step', 'multistep', 'cosine'}.")



def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_update(DEFAULT_CONFIG, user_cfg)
    cfg['config_path'] = os.path.abspath(config_path)
    _validate_config(cfg)
    return cfg



def save_config_copy(cfg: Dict[str, Any], save_path: str) -> None:
    save_obj = copy.deepcopy(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(save_obj, f, sort_keys=False, allow_unicode=True)
