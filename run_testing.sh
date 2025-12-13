#!/bin/bash
# Testing wrapper script for HAPPO MAPush models
# Ensures clean Python path environment

# Unset any PYTHONPATH pollution
unset PYTHONPATH

# Change to project directory
cd /home/gvlab/new-universal-MAPush

# Run testing with clean environment
conda run -n mapush python HARL/harl_mapush/test.py "$@"
