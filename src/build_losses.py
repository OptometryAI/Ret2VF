import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMultiOutputRegressionLoss(nn.Module):
    def __init__(self, loss_type='mse', reduction='mean', delta=1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.delta = delta

        if self.loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif self.loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=delta)
        elif self.loss_type == 'log_cosh':
            self.base_loss = None
        else:
            raise ValueError(f'Unsupported regression loss_type: {loss_type}')

    @staticmethod
    def _log_cosh_loss(predictions, targets):
        return torch.log(torch.cosh(predictions - targets))

    @staticmethod
    def create_weights_from_targets(targets, high_weight=5.0, low_weight=1.0, threshold=20.0):
        return torch.where(
            targets < threshold,
            torch.full_like(targets, float(high_weight)),
            torch.full_like(targets, float(low_weight)),
        )

    def forward(self, predictions, targets, use_weight=False, high_weight=5.0, low_weight=1.0, threshold=20.0):
        if predictions.shape != targets.shape:
            raise ValueError(f'Shape mismatch: {predictions.shape} vs {targets.shape}')

        if self.loss_type == 'log_cosh':
            element_loss = self._log_cosh_loss(predictions, targets)
        else:
            element_loss = self.base_loss(predictions, targets)

        if use_weight:
            weights = self.create_weights_from_targets(
                targets,
                high_weight=high_weight,
                low_weight=low_weight,
                threshold=threshold,
            )
            element_loss = element_loss * weights

        sample_loss = element_loss.sum(dim=1)
        if self.reduction == 'mean':
            return sample_loss.mean()
        if self.reduction == 'sum':
            return sample_loss.sum()
        return sample_loss


def create_importance_weights(levels, high_weight=5.0, low_weight=1.0, threshold=20.0):
    level_sums = torch.sum(levels, dim=-1)
    weights = torch.where(
        level_sums < threshold,
        torch.full_like(level_sums, float(high_weight)),
        torch.full_like(level_sums, float(low_weight)),
    )
    return weights.unsqueeze(-1).expand_as(levels)


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, levels, use_weight=False, high_weight=5.0, low_weight=1.0, threshold=20.0):
        logits = logits.float()
        levels = levels.float()

        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)
        loss = -(levels * log_p + (1.0 - levels) * log_not_p)

        if use_weight:
            imp = create_importance_weights(
                levels,
                high_weight=high_weight,
                low_weight=low_weight,
                threshold=threshold,
            )
            loss = loss * imp

        loss = loss.sum(dim=-1).sum(dim=-1)  # [batch, channels, classes] -> [batch]
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        pt = probs[batch_indices, targets]

        alpha_t = torch.ones_like(pt, dtype=torch.float32)
        alpha_t = torch.where(targets == 0, torch.full_like(alpha_t, self.alpha), torch.full_like(alpha_t, 1 - self.alpha))

        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        focal_weight = focal_weight / (focal_weight.mean() + 1e-8)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def build_loss_functions(cfg):
    task_cfg = cfg['task']
    loss_cfg = cfg['loss']

    if task_cfg['task_type'] == 'r':
        main_loss_fn = WeightedMultiOutputRegressionLoss(
            loss_type=loss_cfg.get('main_loss', 'mse'),
            reduction=loss_cfg.get('main_reduction', 'mean'),
        )
    elif task_cfg['task_type'] == 'c':
        main_loss_fn = OrdinalRegressionLoss(
            reduction=loss_cfg.get('main_reduction', 'mean'),
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_cfg['task_type']}")

    bundle = {'main_loss_fn': main_loss_fn}

    if task_cfg.get('predict_classes', True):
        bundle['pm_cls_loss_fn'] = FocalLoss(
            alpha=loss_cfg.get('pm_focal_alpha', 0.7),
            gamma=loss_cfg.get('pm_focal_gamma', 2.0),
            reduction='mean',
        )
        bundle['md_reg_loss_fn'] = nn.MSELoss()

    return bundle
