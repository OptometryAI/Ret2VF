from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import Any, Dict


def create_save_path(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    save_cfg = cfg.get('save', {})
    save_root = save_cfg.get('save_root', 'save_result')
    task_name = save_cfg.get('task_name', cfg.get('experiment_name', 'default_experiment'))
    experiment_name = cfg.get('experiment_name', 'exp')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_path = os.path.join(save_root, task_name, f'{experiment_name}_{timestamp}')
    os.makedirs(save_path, exist_ok=True)

    cfg['save_path'] = save_path
    cfg['timestamp'] = timestamp
    return cfg
