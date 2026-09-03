#!/bin/bash
set -e
apt-get update -qq
apt-get install -y -qq wget bzip2 >/dev/null 2>&1
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda
export PATH="/opt/miniconda/bin:$PATH"
conda install -y --override-channels -c conda-forge spot pandas >/dev/null 2>&1
pip install -q huggingface_hub pyarrow
python3 /ws/ltl_safety_check.py
