#!/bin/bash

log_dir="run_log"

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    echo "Created log directory: $log_dir"
fi

log_file="preprocess.log"

echo "Activating conda environment: LipFD"
source /home/yanyanhao00/miniconda3/etc/profile.d/conda.sh
conda activate LipFD

nohup python preprocess.py > "$log_file" 2>&1 &


