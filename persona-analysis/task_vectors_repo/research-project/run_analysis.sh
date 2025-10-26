#!/bin/bash
# Script to run the analysis in stages to manage memory usage

# Step 1: Set GPU ID (adjust as needed)
export CUDA_VISIBLE_DEVICES=0

# Step 2: Clear GPU memory and OS cache 
echo "Clearing cache..."
# Uncomment the following if you have sudo access
# sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# Step 3: Run with memory tracking
echo "Running analysis..."
python3 -u main.py
