#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="results/partb_horizon_martial"
mkdir -p "${OUT_DIR}"

HORIZONS=(5 10 20 40 80)

for H in "${HORIZONS[@]}"; do
  echo "======================================"
  echo "Part B horizon evaluation: ${H} frames"
  echo "======================================"

  python code/evaluate_representation.py \
    --representation pos \
    --dances_folder train_data_pos/martial/ \
    --read_weight_path weights_pos_stage100k/0100000.weight \
    --in_frame 171 \
    --out_frame 171 \
    --hidden_size 1024 \
    --batch_size 8 \
    --initial_seq_len 20 \
    --eval_frames "${H}" \
    --trials 20 \
    --output_json "${OUT_DIR}/pos_h${H}.json"

  python code/evaluate_representation.py \
    --representation euler \
    --dances_folder train_data_euler/martial/ \
    --read_weight_path weights_euler_stage100k/0100000.weight \
    --in_frame 132 \
    --out_frame 132 \
    --hidden_size 1024 \
    --batch_size 8 \
    --initial_seq_len 20 \
    --eval_frames "${H}" \
    --trials 20 \
    --output_json "${OUT_DIR}/euler_h${H}.json"

  python code/evaluate_representation.py \
    --representation quad \
    --dances_folder train_data_quad/martial/ \
    --read_weight_path weights_quad_stage100k/0100000.weight \
    --in_frame 175 \
    --out_frame 175 \
    --hidden_size 1024 \
    --batch_size 8 \
    --initial_seq_len 20 \
    --eval_frames "${H}" \
    --trials 20 \
    --output_json "${OUT_DIR}/quad_h${H}.json"
done

echo "Part B horizon evaluation complete."
