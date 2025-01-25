#!/bin/bash
# pip install virtualenv
# source /path/to/venv/bin/activate  # Activate virtual environment
# pip install -r requirements_g.txt
conda activate general
python generate_benchmarks/setup.py configs/setup_config_vnncomp.yaml
