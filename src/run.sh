#!/bin/bash

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

log_dir="run_log"

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    echo "Created log directory: $log_dir"
fi

log_file="${log_dir}/train_${timestamp}.log"

echo "Activating conda environment: LipFD"
source /home/yanyanhao00/miniconda3/etc/profile.d/conda.sh
conda activate LipFD

nohup python train.py > "$log_file" 2>&1 &

echo "Training job started. Logs are being saved to $log_file"
