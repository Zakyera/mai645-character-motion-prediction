#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-stage1}"
ITERATION="${2:-0000000}"

OUT_DIR="results/${TAG}_eval"
mkdir -p "${OUT_DIR}"

echo "======================================"
echo "Evaluating all representations"
echo "TAG=${TAG}"
echo "ITERATION=${ITERATION}"
echo "OUT_DIR=${OUT_DIR}"
echo "======================================"

python code/evaluate_representation.py \
  --representation pos \
  --dances_folder train_data_pos/martial/ \
  --read_weight_path weights_pos_${TAG}/${ITERATION}.weight \
  --in_frame 171 \
  --out_frame 171 \
  --hidden_size 1024 \
  --batch_size 8 \
  --initial_seq_len 20 \
  --eval_frames 20 \
  --trials 20 \
  --output_json "${OUT_DIR}/pos.json"

python code/evaluate_representation.py \
  --representation euler \
  --dances_folder train_data_euler/martial/ \
  --read_weight_path weights_euler_${TAG}/${ITERATION}.weight \
  --in_frame 132 \
  --out_frame 132 \
  --hidden_size 1024 \
  --batch_size 8 \
  --initial_seq_len 20 \
  --eval_frames 20 \
  --trials 20 \
  --output_json "${OUT_DIR}/euler.json"

python code/evaluate_representation.py \
  --representation quad \
  --dances_folder train_data_quad/martial/ \
  --read_weight_path weights_quad_${TAG}/${ITERATION}.weight \
  --in_frame 175 \
  --out_frame 175 \
  --hidden_size 1024 \
  --batch_size 8 \
  --initial_seq_len 20 \
  --eval_frames 20 \
  --trials 20 \
  --output_json "${OUT_DIR}/quad.json"

echo "Done evaluation for ${TAG}"
