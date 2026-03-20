import argparse
import os

from build_dataset import make_dataloaders
from build_model import generate_model
from build_losses import build_loss_functions
from build_metrics import build_metric_functions
from build_optimizer import build_optimizer_and_scheduler
from engine import fit
from utils.config import load_config, save_config_copy
from utils.create_save_path import create_save_path
from utils.logger import ExperimentLogger
from utils.seed import set_seed


def parse_cli():
    parser = argparse.ArgumentParser(description='OCT2VF training entry')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    return parser.parse_args()


def main():
    cli_args = parse_cli()
    cfg = load_config(cli_args.config)

    # Create the run directory and persist the resolved runtime config.
    cfg = create_save_path(cfg)
    save_config_copy(cfg, os.path.join(cfg['save_path'], 'resolved_config.yaml'))

    # Reproducibility settings.
    reproducibility_cfg = cfg.get('reproducibility', {})
    set_seed(
        cfg['seed'],
        deterministic=reproducibility_cfg.get('deterministic', True),
        warn_only=reproducibility_cfg.get('warn_only', True),
    )

    # logger / tensorboard
    logger = ExperimentLogger(
        save_path=cfg['save_path'],
        use_tensorboard=cfg.get('logging', {}).get('use_tensorboard', True),
        use_test=cfg.get('train', {}).get('use_test', True),
    )

    # dataloaders / model / optim / losses / metrics
    train_loader, val_loader, test_loader = make_dataloaders(cfg)
    model = generate_model(cfg).to(cfg['device'])
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    loss_fns = build_loss_functions(cfg)
    metric_fns = build_metric_functions(cfg)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fns=loss_fns,
        metric_fns=metric_fns,
        cfg=cfg,
        logger=logger,
    )

    logger.close()


if __name__ == '__main__':
    main()
