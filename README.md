# OCT2VF: Multi-Task Visual Field Prediction from 3D OCT Volumes

## Overview
This repository provides a training pipeline for predicting 54-point visual field (VF) values from 3D optical coherence tomography (OCT) volumes. The framework supports a primary VF prediction objective and two auxiliary tasks for multi-task learning.

Primary task:
- VF prediction (`task_type: r` for direct regression, `task_type: c` for ordinal regression)

Auxiliary tasks:
- Pathology meta-label classification (PM binary classification)
- Mean Deviation (MD) regression

The main training entry point is [`src/train.py`](src/train.py).

## Repository Structure
```text
OCT2VF/
├── configs/
│   └── OR+aux.yaml
├── data/
│   └── processed_oct_npy/
├── scripts/
│   └── OR+aux.sh
├── src/
│   ├── train.py
│   ├── engine.py
│   ├── build_dataset.py
│   ├── build_model.py
│   ├── build_losses.py
│   ├── build_metrics.py
│   ├── build_optimizer.py
│   ├── dataframe_dealer.py
│   └── utils/
├── data_table_with_processed_OCT.xlsx
└── save_result/
```

## Environment
Recommended Python version: 3.10+

```bash
pip install -r requirements.txt
pip install openpyxl tensorboard
```

Notes:
- `openpyxl` is required by `pandas.read_excel` for `.xlsx` files.
- `tensorboard` is required for experiment visualization.

## Data Specification
The default metadata table is:

```text
data_table_with_processed_OCT.xlsx
```

Expected columns used by the pipeline:
- `processed_oct_path`: path to the preprocessed OCT `.npy` file
- `HVF_VisualFieldPlot`: serialized VF grid string to be parsed into 54 VF points
- `Laterality`: eye laterality (`R` is mirrored for left-right alignment)
- `Meta-PM`: pathology group label (`C0` to `C4`, missing values allowed)
- `HVF_MD`: MD value for auxiliary regression
- `PID`, `Age`: sample metadata returned with each batch

## One-Command Training
```bash
bash scripts/OR+aux.sh
```

Equivalent direct command:
```bash
python src/train.py --config configs/OR+aux.yaml
```

## Core Configuration
Default config: [`configs/OR+aux.yaml`](configs/OR+aux.yaml)

Important fields:
- `task.task_type`: `c` (ordinal regression) or `r` (direct regression)
- `task.predict_classes`: enable/disable auxiliary PM and MD branches
- `data.dataset_ratio`: global split ratio `[train, val, test]`
- `loss.pm_loss_weight`, `loss.md_loss_weight`: auxiliary loss weights
- `optimizer`, `scheduler`: optimization and LR scheduling policies
- `selection.monitor`: validation metric for best-checkpoint selection (default: `rmse`)

## Data Splitting Strategy
The split is not purely random. The implemented strategy is:
1. All samples with non-empty `Meta-PM` labels are assigned to training to preserve supervision for PM classification.
2. Samples with empty `Meta-PM` labels are stratified by MD severity (`Mild`, `Moderate`, `Severe`).
3. The final train/val/test partitions are constructed from these stratified subsets according to `dataset_ratio`.

As a consequence, PM classification metrics may be informative in training but limited in validation/test when PM labels are unavailable.

## Outputs and Logging
Each run creates:

```text
save_result/<task_name>/<experiment_name>_<timestamp>/
```

Typical artifacts:
- `resolved_config.yaml`: fully resolved runtime config
- `epochs_record.txt`: per-epoch train/val/test metrics
- `best_record.txt`: best-metric update history
- `best_model.pth`: best checkpoint (by `selection.monitor`)
- `last_model.pth`: final epoch checkpoint
- `checkpoint_epoch_*.pth`: periodic checkpoints (`train.checkpoint`)
- `tensorboard/`: TensorBoard logs

Launch TensorBoard:
```bash
tensorboard --logdir save_result
```

## Model and Objective Design
Backbone:
- 3D ResNet (`model_depth` in `{10, 18, 34, 50, 101, 152, 200}`)

Primary output:
- `r`: direct regression for 54 VF points
- `c`: ordinal target encoding with ordered BCE-style optimization

Auxiliary outputs:
- PM binary classification (Focal Loss)
- MD regression (MSE Loss)

Total objective:

```text
total_loss = main_loss + pm_loss_weight * pm_loss + md_loss_weight * md_loss
```

## Reproducibility
The pipeline supports deterministic behavior through seed control and cuDNN deterministic settings (see `reproducibility` in the config and `src/utils/seed.py`).

## Citation and Usage
If you use this codebase in academic work, please cite your associated manuscript or project report and describe the exact configuration and data split protocol for reproducibility.
