#!/bin/bash

source activate unicorrn


export BENCHMARK_CONFIG_PATH="/your_project_path/configs/benchmarks/3d3d.yml"
export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage2.yml"
export CKPT_PATH="/your_project_path/pretrained_models/UniCorrn_Large_Stage2.pth"
export EXP_NAME="UniCorrn_Large_Stage2"

python -m benchmarks.evaluate_3d3d \
    --model_config $MODEL_CONFIG_PATH \
    --benchmark_config $BENCHMARK_CONFIG_PATH \
    --split test_3dmatch \
    --ckpt_path $CKPT_PATH \
    --exp_name $EXP_NAME

python -m benchmarks.evaluate_3d3d \
    --model_config $MODEL_CONFIG_PATH \
    --benchmark_config $BENCHMARK_CONFIG_PATH \
    --split test_3dlomatch \
    --ckpt_path $CKPT_PATH \
    --exp_name $EXP_NAME
