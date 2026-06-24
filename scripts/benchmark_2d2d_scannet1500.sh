#!/bin/bash

source activate unicorrn

export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage2.yml"
export CKPT_PATH="/your_project_path/pretrained_models/UniCorrn_Large_Stage2.pth"
export EXP_NAME="UniCorrn_Large_Stage2"
export QUERY_POINTS_PATH="/your_project_path/benchmarks/scannet1500_query_points.json"

python -m benchmarks.evaluate_2d2d \
    --model_config $MODEL_CONFIG_PATH \
    --ckpt_path $CKPT_PATH \
    --query_points_path $QUERY_POINTS_PATH \
    --benchmark "scannet_1500" \
    --coarse_coverage 0.9 \
    --overlap 0.50 \
    --exp_name $EXP_NAME \
    --unified_model
