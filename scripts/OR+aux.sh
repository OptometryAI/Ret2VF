#!/bin/bash
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python src/train.py --config configs/OR+aux.yaml
