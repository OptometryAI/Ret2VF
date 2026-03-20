import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, save_path: str, use_tensorboard: bool = True, use_test: bool = True):
        self.save_path = save_path
        self.use_tensorboard = use_tensorboard
        self.use_test = use_test

        self.record_file = os.path.join(save_path, 'epochs_record.txt')
        self.best_file = os.path.join(save_path, 'best_record.txt')

        self.train_writer: Optional[SummaryWriter] = None
        self.val_writer: Optional[SummaryWriter] = None
        self.test_writer: Optional[SummaryWriter] = None

        if self.use_tensorboard:
            self.train_writer = SummaryWriter(os.path.join(save_path, 'tensorboard', 'train'))
            self.val_writer = SummaryWriter(os.path.join(save_path, 'tensorboard', 'val'))
            if self.use_test:
                self.test_writer = SummaryWriter(os.path.join(save_path, 'tensorboard', 'test'))

    @staticmethod
    def _format_metrics(metrics):
        return ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])

    def _write_tensorboard(self, writer: Optional[SummaryWriter], phase: str, metrics: dict, epoch: int):
        if writer is None:
            return

        writer.add_scalars('Loss', {
            'Main': metrics.get('main_loss', 0.0),
            'PM_Classification': metrics.get('pm_cls_loss', 0.0),
            'MD_Regression': metrics.get('md_reg_loss', 0.0),
            'Total': metrics.get('total_loss', 0.0),
        }, epoch)

        writer.add_scalars('VF_Metrics', {
            'RMSE': metrics.get('rmse', 0.0),
            'MAE': metrics.get('mae', 0.0),
        }, epoch)

        if 'pm_cls_acc' in metrics:
            writer.add_scalar('PM_Cls_Accuracy', metrics['pm_cls_acc'], epoch)
        if 'md_rmse' in metrics or 'md_mae' in metrics:
            writer.add_scalars('MD_Metrics', {
                'RMSE': metrics.get('md_rmse', 0.0),
                'MAE': metrics.get('md_mae', 0.0),
            }, epoch)

    def log_lr(self, epoch: int, lr: float):
        if self.train_writer is not None:
            self.train_writer.add_scalar('LR', lr, epoch)

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, test_metrics: Optional[dict] = None):
        train_line = f"Train: {self._format_metrics(train_metrics)}"
        val_line = f"Val: {self._format_metrics(val_metrics)}"
        print(f'Epoch {epoch + 1}')
        print(train_line)
        print(val_line)

        if test_metrics is not None:
            test_line = f"Test: {self._format_metrics(test_metrics)}"
            print(test_line)
        else:
            test_line = None

        with open(self.record_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch + 1}\n')
            f.write(train_line + '\n')
            f.write(val_line + '\n')
            if test_line is not None:
                f.write(test_line + '\n')
            f.write('\n')

        self._write_tensorboard(self.train_writer, 'train', train_metrics, epoch)
        self._write_tensorboard(self.val_writer, 'val', val_metrics, epoch)
        if test_metrics is not None:
            self._write_tensorboard(self.test_writer, 'test', test_metrics, epoch)

    def log_best(self, epoch: int, best_score: float, monitor: str):
        line = f'Best updated at epoch {epoch + 1}, {monitor}: {best_score:.6f}'
        print(line)
        with open(self.best_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def close(self):
        for writer in [self.train_writer, self.val_writer, self.test_writer]:
            if writer is not None:
                writer.close()
