#!/bin/bash

set -e

conda_env_name="cv_homework"

conda create -n $conda_env_name python=3.11

conda activate $conda_env_name

# Install Python Package by pip
python -m pip install -r requirements.txt

# Install Playwright Browser
#python -m playwright install