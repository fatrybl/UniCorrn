#!/bin/bash

module load cuda/12.8.0
source activate unicorrn

export BENCHMARK_CONFIG_PATH="/your_project_path/configs/benchmarks/3d3d.yml"
export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage2.yml"
export CKPT_PATH="/your_project_path/pretrained_models/UniCorrn_Large_Stage2.pth"
export EXP_NAME="UniCorrn_Large_Stage2"

python -m benchmarks.evaluate_3d3d_modelnet \
    --model_config $MODEL_CONFIG_PATH \
    --benchmark_config $BENCHMARK_CONFIG_PATH \
    --ckpt_path $CKPT_PATH \
    --exp_name $EXP_NAME
