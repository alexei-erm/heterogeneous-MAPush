#!/bin/bash
# Training wrapper that ensures clean Python path

# Unset any PYTHONPATH pollution
unset PYTHONPATH

# Run training with clean environment and unbuffered output
cd /home/gvlab/new-universal-MAPush
conda run -n mapush --no-capture-output python -u HARL/harl_mapush/train.py "$@"
