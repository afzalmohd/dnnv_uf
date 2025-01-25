#!/bin/bash
# pip3 install virtualenv
# virtualenv env_abcrown
# source env_abcrown/bin/activate  # Activate virtual environment
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114
# pip3 install --trusted-host rni.tcsapps.com torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114
# pip3 install --proxy=http://proxy.tcs.com:8080 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114

# python3 info.py
# pip3 install -r requirements_ab.txt
conda activate alpha-beta-crown
python abcrown_tool_run_scripts/script.py configs/setup_config_vnncomp.yaml
