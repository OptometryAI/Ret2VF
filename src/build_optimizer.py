from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)



def build_optimizer(cfg, model) -> Optimizer:
    opt_cfg = cfg['optimizer']
    name = opt_cfg.get('name', 'adam').lower()
    lr = float(opt_cfg.get('lr', 1e-4))
    weight_decay = float(opt_cfg.get('weight_decay', 0.0))

    params = model.parameters()

    if name == 'adam':
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
            eps=float(opt_cfg.get('eps', 1e-8)),
        )
    if name == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
            eps=float(opt_cfg.get('eps', 1e-8)),
        )
    if name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(opt_cfg.get('momentum', 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(opt_cfg.get('nesterov', False)),
        )

    raise ValueError(f'Unsupported optimizer name: {name}')



def build_scheduler(cfg, optimizer: Optimizer) -> Optional[object]:
    sched_cfg = cfg.get('scheduler', {})
    name = sched_cfg.get('name', 'none').lower()

    if name == 'none':
        return None

    if name == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get('mode', 'min'),
            factor=float(sched_cfg.get('factor', 0.5)),
            patience=int(sched_cfg.get('patience', 2)),
            threshold=float(sched_cfg.get('threshold', 1e-4)),
            min_lr=float(sched_cfg.get('min_lr', 0.0)),
            verbose=bool(sched_cfg.get('verbose', True)),
        )

    if name == 'step':
        return StepLR(
            optimizer,
            step_size=int(sched_cfg.get('step_size', 10)),
            gamma=float(sched_cfg.get('gamma', 0.1)),
        )

    if name == 'multistep':
        milestones = sched_cfg.get('milestones', [30, 60, 90])
        return MultiStepLR(
            optimizer,
            milestones=list(milestones),
            gamma=float(sched_cfg.get('gamma', 0.1)),
        )

    if name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get('t_max', cfg['train']['epochs'])),
            eta_min=float(sched_cfg.get('eta_min', 0.0)),
        )

    raise ValueError(f'Unsupported scheduler name: {name}')



def build_optimizer_and_scheduler(cfg, model) -> Tuple[Optimizer, Optional[object]]:
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    return optimizer, scheduler
