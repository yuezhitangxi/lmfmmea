#!/usr/bin/env bash
set -euo pipefail

dataset='FBDB15K'
ratio=0.2
seed=2023
file_name="m5_smoke_${dataset}_${ratio}"

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

"${PYTHON_BIN}" -u src/run.py \
    --device cpu \
    --file_dir data/mmkg/${dataset}/norm \
    --pred_name ${file_name} \
    --rate ${ratio} \
    --lr .0005 \
    --epochs 1 \
    --dropout 0.0 \
    --hidden_units "300,300,300" \
    --check_point 1 \
    --bsize 3500 \
    --il_start 50 \
    --semi_learn_step 5 \
    --csls \
    --csls_k 3 \
    --seed ${seed} \
    --tau 0.1 \
    --tau2 4.0 \
    --structure_encoder "Dualmodal-joint-LMF" \
    --joint_type 1 \
    --img_dim 300 \
    --attr_dim 300 \
    --name_dim 100 \
    --char_dim 100 \
    --bi_adapter \
    --adapter_choice 0 \
    --use_ms_loss \
    --use_joint_loss \
    --joint_use_nce \
    --use_cosface_loss \
    --cosface_margin 0.15 \
    --cosface_scale 2 \
    --cosface_hard_topk 10 \
    --cosface_hard_weight 0.05 \
    --cosface_hard_margin 0.15 \
    --cosface_focal_gamma 0 \
    --cosface_t_max 0.1 \
    --cosface_warmup_epoch 50 \
    --LMFrank 8 \
    --use_GphForward 1 \
    --add_other_modal 0 \
    --Is_LMFSoftmax 0 \
    --mr_fusion_type 7 \
    --final_fusion_type 0 \
    --use_proxy \
    --w_name \
    --w_char \
    2>&1 | tee logs/${file_name}.log
