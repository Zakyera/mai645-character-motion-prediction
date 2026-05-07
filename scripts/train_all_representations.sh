#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${1:-1000}"
TAG="${2:-stage1}"

BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
SEQ_LEN="${SEQ_LEN:-100}"

mkdir -p logs

echo "======================================"
echo "Training all representations"
echo "ITERATIONS=${ITERATIONS}"
echo "TAG=${TAG}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "HIDDEN_SIZE=${HIDDEN_SIZE}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "======================================"

echo "[1/3] Positional training"
python code/pytorch_train_pos_aclstm.py \
  --dances_folder train_data_pos/martial/ \
  --write_weight_folder weights_pos_${TAG}/ \
  --write_bvh_motion_folder bvh_pos_${TAG}/ \
  --in_frame 171 \
  --out_frame 171 \
  --batch_size "${BATCH_SIZE}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --seq_len "${SEQ_LEN}" \
  --total_iterations "${ITERATIONS}" \
  2>&1 | tee logs/train_pos_${TAG}.log

echo "[2/3] Euler training"
python code/pytorch_train_euler_aclstm.py \
  --dances_folder train_data_euler/martial/ \
  --write_weight_folder weights_euler_${TAG}/ \
  --write_bvh_motion_folder bvh_euler_${TAG}/ \
  --in_frame 132 \
  --out_frame 132 \
  --batch_size "${BATCH_SIZE}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --seq_len "${SEQ_LEN}" \
  --total_iterations "${ITERATIONS}" \
  2>&1 | tee logs/train_euler_${TAG}.log

echo "[3/3] Quaternion training"
python code/pytorch_train_quad_aclstm.py \
  --dances_folder train_data_quad/martial/ \
  --write_weight_folder weights_quad_${TAG}/ \
  --write_bvh_motion_folder bvh_quad_${TAG}/ \
  --in_frame 175 \
  --out_frame 175 \
  --batch_size "${BATCH_SIZE}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --seq_len "${SEQ_LEN}" \
  --total_iterations "${ITERATIONS}" \
  2>&1 | tee logs/train_quad_${TAG}.log

echo "Done training all representations for ${TAG}"
