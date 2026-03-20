from contextlib import nullcontext
import os
from typing import Optional

import torch
from tqdm import tqdm


def _parse_batch(batch_data, cfg):
    device = cfg['device']
    task_type = cfg['task']['task_type']
    predict_classes = cfg['task'].get('predict_classes', True)

    batch = {
        'oct_img': batch_data['oct_3D'].to(device),
        'vf_targets': batch_data['VF_tensor'].to(device).float(),
    }

    if task_type == 'c':
        batch['vf_encoded'] = batch_data['VF_encoded_tensor'].to(device).float()

    if predict_classes:
        batch['pm_targets'] = batch_data['pm_label'].to(device).long()
        batch['md_targets'] = batch_data['MD'].to(device).float()
        batch['valid_pm_mask'] = batch['pm_targets'] != -1

    return batch


def _forward_model(model, oct_img, predict_classes: bool):
    outputs = model(oct_img)

    if predict_classes:
        if not isinstance(outputs, dict):
            raise TypeError('When predict_classes=True, model must return a dict.')
        main_output = outputs['main']
        pm_cls_output = outputs['pm_classification']
        md_reg_output = outputs['md_regression']
        if md_reg_output.ndim > 1 and md_reg_output.shape[-1] == 1:
            md_reg_output = md_reg_output.squeeze(-1)
        return {
            'main': main_output,
            'pm_classification': pm_cls_output,
            'md_regression': md_reg_output,
        }

    if isinstance(outputs, dict):
        return {'main': outputs['main']}
    return {'main': outputs}


def _compute_losses(outputs, batch, loss_fns, cfg):
    task_cfg = cfg['task']
    predict_classes = task_cfg.get('predict_classes', True)
    task_type = task_cfg['task_type']
    loss_cfg = cfg['loss']

    main_loss_fn = loss_fns['main_loss_fn']
    pm_cls_loss_fn = loss_fns.get('pm_cls_loss_fn', None)
    md_reg_loss_fn = loss_fns.get('md_reg_loss_fn', None)

    if task_type == 'r':
        if task_cfg.get('use_weight', False):
            main_loss = main_loss_fn(outputs['main'], batch['vf_targets'], use_weight=True)
        else:
            main_loss = main_loss_fn(outputs['main'], batch['vf_targets'], use_weight=False)
    elif task_type == 'c':
        if task_cfg.get('use_weight', False):
            main_loss = main_loss_fn(outputs['main'], batch['vf_encoded'], use_weight=True)
        else:
            main_loss = main_loss_fn(outputs['main'], batch['vf_encoded'], use_weight=False)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    pm_cls_loss = torch.tensor(0.0, device=cfg['device'])
    md_reg_loss = torch.tensor(0.0, device=cfg['device'])

    if predict_classes:
        valid_pm_mask = batch['valid_pm_mask']
        if valid_pm_mask.any():
            pm_cls_loss = pm_cls_loss_fn(
                outputs['pm_classification'][valid_pm_mask],
                batch['pm_targets'][valid_pm_mask],
            )
        md_reg_loss = md_reg_loss_fn(outputs['md_regression'], batch['md_targets'])

    total_loss = (
        main_loss
        + loss_cfg.get('pm_loss_weight', 0.0) * pm_cls_loss
        + loss_cfg.get('md_loss_weight', 0.0) * md_reg_loss
    )

    return {
        'main_loss': main_loss,
        'pm_cls_loss': pm_cls_loss,
        'md_reg_loss': md_reg_loss,
        'total_loss': total_loss,
    }


def _init_storage():
    return {
        'main_predictions': [],
        'main_targets': [],
        'pm_outputs': [],
        'pm_targets': [],
        'md_outputs': [],
        'md_targets': [],
        'main_loss_sum': 0.0,
        'pm_cls_loss_sum': 0.0,
        'md_reg_loss_sum': 0.0,
        'total_loss_sum': 0.0,
        'num_steps': 0,
    }


def _update_storage(storage, outputs, batch, loss_dict, cfg):
    predict_classes = cfg['task'].get('predict_classes', True)

    storage['main_predictions'].append(outputs['main'].detach().cpu())
    storage['main_targets'].append(batch['vf_targets'].detach().cpu())

    if predict_classes:
        valid_pm_mask = batch['valid_pm_mask']
        if valid_pm_mask.any():
            storage['pm_outputs'].append(outputs['pm_classification'][valid_pm_mask].detach().cpu())
            storage['pm_targets'].append(batch['pm_targets'][valid_pm_mask].detach().cpu())
        storage['md_outputs'].append(outputs['md_regression'].detach().cpu())
        storage['md_targets'].append(batch['md_targets'].detach().cpu())

    storage['main_loss_sum'] += float(loss_dict['main_loss'].item())
    storage['pm_cls_loss_sum'] += float(loss_dict['pm_cls_loss'].item())
    storage['md_reg_loss_sum'] += float(loss_dict['md_reg_loss'].item())
    storage['total_loss_sum'] += float(loss_dict['total_loss'].item())
    storage['num_steps'] += 1


def _finalize_metrics(storage, metric_fns, cfg):
    metrics = {}
    num_steps = max(storage['num_steps'], 1)

    metrics['main_loss'] = storage['main_loss_sum'] / num_steps
    metrics['pm_cls_loss'] = storage['pm_cls_loss_sum'] / num_steps
    metrics['md_reg_loss'] = storage['md_reg_loss_sum'] / num_steps
    metrics['total_loss'] = storage['total_loss_sum'] / num_steps

    main_predictions = torch.cat(storage['main_predictions'], dim=0)
    main_targets = torch.cat(storage['main_targets'], dim=0)
    metrics.update(metric_fns['main_metrics_fn'](main_predictions, main_targets))

    if cfg['task'].get('predict_classes', True):
        if len(storage['pm_outputs']) > 0:
            pm_outputs = torch.cat(storage['pm_outputs'], dim=0)
            pm_targets = torch.cat(storage['pm_targets'], dim=0)
            metrics.update(metric_fns['pm_metrics_fn'](pm_outputs, pm_targets))
        else:
            metrics['pm_cls_acc'] = 0.0

        if len(storage['md_outputs']) > 0:
            md_outputs = torch.cat(storage['md_outputs'], dim=0)
            md_targets = torch.cat(storage['md_targets'], dim=0)
            metrics.update(metric_fns['md_metrics_fn'](md_outputs, md_targets))
        else:
            metrics['md_rmse'] = 0.0
            metrics['md_mae'] = 0.0

    return metrics


def run_one_epoch(
    model,
    loader,
    optimizer,
    loss_fns,
    metric_fns,
    cfg,
    phase: str,
):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()

    storage = _init_storage()
    context = nullcontext() if is_train else torch.no_grad()

    for batch_data in tqdm(loader, desc=f'{phase}_epoch', leave=False):
        batch = _parse_batch(batch_data, cfg)

        if is_train:
            optimizer.zero_grad()

        with context:
            outputs = _forward_model(model, batch['oct_img'], cfg['task'].get('predict_classes', True))
            loss_dict = _compute_losses(outputs, batch, loss_fns, cfg)

            if is_train:
                loss_dict['total_loss'].backward()
                max_grad_norm = cfg.get('train', {}).get('max_grad_norm', None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        _update_storage(storage, outputs, batch, loss_dict, cfg)

    return _finalize_metrics(storage, metric_fns, cfg)


def evaluate(model, loader, loss_fns, metric_fns, cfg, phase='val'):
    return run_one_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        loss_fns=loss_fns,
        metric_fns=metric_fns,
        cfg=cfg,
        phase=phase,
    )


def fit(
    model,
    train_loader,
    val_loader,
    test_loader: Optional[object],
    optimizer,
    scheduler,
    loss_fns,
    metric_fns,
    cfg,
    logger,
):
    best_score = float('inf')
    epochs = cfg['train']['epochs']
    use_test = cfg['train'].get('use_test', True) and test_loader is not None
    best_metric_name = cfg.get('selection', {}).get('monitor', 'rmse')
    scheduler_mode = cfg.get('scheduler', {}).get('step_on', 'val_main_loss')

    for epoch in range(epochs):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fns=loss_fns,
            metric_fns=metric_fns,
            cfg=cfg,
            phase='train',
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fns=loss_fns,
            metric_fns=metric_fns,
            cfg=cfg,
            phase='val',
        )

        test_metrics = None
        if use_test:
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                loss_fns=loss_fns,
                metric_fns=metric_fns,
                cfg=cfg,
                phase='test',
            )

        # By default, drive LR scheduling with validation main loss.
        if scheduler is not None:
            if scheduler_mode == 'val_main_loss':
                scheduler.step(val_metrics['main_loss'])
            elif scheduler_mode == 'train_main_loss':
                scheduler.step(train_metrics['main_loss'])
            else:
                scheduler.step(val_metrics['main_loss'])

        logger.log_epoch(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics, test_metrics=test_metrics)
        logger.log_lr(epoch=epoch, lr=optimizer.param_groups[0]['lr'])

        current_score = val_metrics.get(best_metric_name, val_metrics['rmse'])
        if current_score < best_score:
            best_score = current_score
            best_path = os.path.join(cfg['save_path'], 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            logger.log_best(epoch=epoch, best_score=best_score, monitor=best_metric_name)

        checkpoint_every = cfg['train'].get('checkpoint', 0)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(cfg['save_path'], f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)

    last_path = os.path.join(cfg['save_path'], 'last_model.pth')
    torch.save(model.state_dict(), last_path)
