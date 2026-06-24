#!/bin/bash

source activate unicorrn

export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage2.yml"
export CKPT_PATH="/your_project_path/pretrained_models/UniCorrn_Large_Stage2.pth"
export EXP_NAME="UniCorrn_Large_Stage2"

python -m benchmarks.inloc_benchmark \
    --model_config $MODEL_CONFIG_PATH \
    --ckpt_path $CKPT_PATH \
    --exp_name $EXP_NAME \
    --dataset "VislocInLoc('data/Datasets/InLoc/InLoc_wo_images', pairsfile='pairs-query-netvlad40-temporal', topk=40)" \
    --max_image_size 1200 \
    --confidence_threshold "-5" \
    --matching_radius_px 2.5 \
    --max_keypoints 5000 \
    --reprojection_error_diag_ratio 0.008 \
    --run_id $run_id
