import torch
import torch.nn.functional as F


def decode_ordinal_predictions(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    levels = (probs > threshold).float()
    preds = torch.sum(levels, dim=-1)
    return preds


def compute_regression_metrics(predictions, targets):
    predictions = torch.round(predictions)
    mae = F.l1_loss(predictions, targets).item()
    mse = F.mse_loss(predictions, targets).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return {
        'rmse': rmse,
        'mae': mae,
    }


def compute_ordinal_metrics(logits, targets, threshold=0.5):
    preds = decode_ordinal_predictions(logits, threshold=threshold)
    mae = F.l1_loss(preds.float(), targets.float()).item()
    mse = F.mse_loss(preds.float(), targets.float()).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return {
        'rmse': rmse,
        'mae': mae,
    }


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def compute_pm_metrics(outputs, targets):
    return {'pm_cls_acc': calculate_accuracy(outputs, targets)}


def compute_md_metrics(outputs, targets):
    outputs = outputs.float()
    targets = targets.float()
    rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
    mae = torch.mean(torch.abs(outputs - targets)).item()
    return {
        'md_rmse': rmse,
        'md_mae': mae,
    }


def build_metric_functions(cfg):
    task_cfg = cfg['task']
    metric_cfg = cfg.get('metrics', {})
    ordinal_threshold = metric_cfg.get('ordinal_threshold', 0.5)

    if task_cfg['task_type'] == 'r':
        main_metrics_fn = compute_regression_metrics
    elif task_cfg['task_type'] == 'c':
        def main_metrics_fn(logits, targets):
            return compute_ordinal_metrics(logits, targets, threshold=ordinal_threshold)
    else:
        raise ValueError(f"Unsupported task_type: {task_cfg['task_type']}")

    return {
        'main_metrics_fn': main_metrics_fn,
        'pm_metrics_fn': compute_pm_metrics,
        'md_metrics_fn': compute_md_metrics,
    }
