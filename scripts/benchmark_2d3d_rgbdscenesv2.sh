#!/bin/bash

source activate unicorrn

export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage2.yml"
export CKPT_PATH="/your_project_path/pretrained_models/UniCorrn_Large_Stage2.pth"
export EXP_NAME="UniCorrn_Large_Stage2"

python -m benchmarks.evaluate_2d3d_rgbdscenes \
    --model_config $MODEL_CONFIG_PATH \
    --ckpt_path $CKPT_PATH \
    --dataset_dir "data/Datasets/RGBDScenesV2/" \
    --split test \
    --exp_name $EXP_NAME
