#!/bin/bash
# Fixed training command for VARS-Training
# This addresses view_mask issues and training stability problems

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_lightning.py \
    --dataset_root . \
    --backbone_type mvit \
    --backbone_name mvit_base_16x4 \
    --frames_per_clip 16 \
    --start_frame 40 \
    --end_frame 82 \
    --clips_per_video 3 \
    --clip_sampling uniform \
    --batch_size 2 \
    --accumulate_grad_batches 16 \
    --num_workers 4 \
    --max_views 5 \
    --loss_function focal \
    --use_class_balanced_sampler \
    --oversample_factor 2.5 \
    --class_weighting_strategy none \
    --freezing_strategy adaptive \
    --adaptive_patience 2 \
    --adaptive_min_improvement 0.001 \
    --lr 5e-4 \
    --backbone_lr 5e-5 \
    --gradient_clip_norm 0.5 \
    --weight_decay 0.01 \
    --epochs 40 \
    --seed 42 