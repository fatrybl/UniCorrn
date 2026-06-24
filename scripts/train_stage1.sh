#!/bin/bash

source activate unicorrn

export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage1.yml"
export TRAINER_CONFIG_PATH="/your_project_path/configs/trainers/trainer_large_scale_2d2d_3d3d_joint_stage_1.yml"
export CROCO_WEIGHTS_PATH="/your_project_path/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"
export CROCO_DECODER_PATH="/your_project_path/pretrained_models/CroCoV2_Large_BaseDecoder.pth"
export OUTPUT_DIR="/your_output_directory/"

accelerate launch --num_processes 4 \
    --multi_gpu \
    --main_process_port 12345 \
    --num_machines 1 \
    /your_project_path/train.py \
    --model_config $MODEL_CONFIG_PATH \
    --trainer_config $TRAINER_CONFIG_PATH \
    --pretrained_croco_weights $CROCO_WEIGHTS_PATH \
    --pretrained_croco_decoder_weights $CROCO_DECODER_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 6 \
    --accum_iter 1 \
    --set_static_graph
