import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import read_bvh


def ensure_trailing_slash(path):
    return path if path.endswith("/") else path + "/"


def load_dances(dance_folder):
    dance_folder = ensure_trailing_slash(dance_folder)
    dances = []

    for name in sorted(os.listdir(dance_folder)):
        if not name.endswith(".npy"):
            continue
        arr = np.load(dance_folder + name)
        dances.append(arr)

    if not dances:
        raise RuntimeError(f"No .npy files found in {dance_folder}")

    return dances


def get_model_class(representation):
    if representation == "pos":
        from synthise_pos_motion import acLSTM
        return acLSTM
    if representation == "euler":
        from synthise_euler_motion import acLSTM
        return acLSTM
    if representation == "quad":
        from synthise_quad_motion import acLSTM
        return acLSTM

    raise ValueError(f"Unknown representation: {representation}")


def get_root_xz_indices(representation):
    if representation == "pos":
        hip_index = read_bvh.joint_index["hip"]
        return hip_index * 3, hip_index * 3 + 2

    # Euler and quaternion representations keep root translation in channels 0:3.
    return 0, 2


def convert_root_xz_to_differences(seq, x_idx, z_idx):
    """Convert root x/z absolute trajectory into frame-to-frame differences.

    This mirrors the training scripts.
    Input shape: [batch, frames, channels]
    Output shape: [batch, frames - 1, channels]
    """
    dif = seq[:, 1:seq.shape[1]] - seq[:, 0:seq.shape[1] - 1]
    out = seq[:, 0:seq.shape[1] - 1].copy()
    out[:, :, x_idx] = dif[:, :, x_idx]
    out[:, :, z_idx] = dif[:, :, z_idx]
    return out


def sample_batch(dances, batch_size, total_len, frame_rate):
    speed = frame_rate / 30
    batch = []

    for _ in range(batch_size):
        dance = random.choice(dances)
        max_start = int(dance.shape[0] - total_len * speed - 10)
        if max_start <= 10:
            raise RuntimeError(
                f"Dance too short for total_len={total_len}, frame_rate={frame_rate}: {dance.shape}"
            )

        start_id = random.randint(10, max_start)

        seq = []
        for i in range(total_len):
            seq.append(dance[int(i * speed + start_id)])

        batch.append(seq)

    return np.array(batch)


def evaluate_once(model, dances, representation, batch_size, initial_seq_len, eval_frames, frame_rate):
    x_idx, z_idx = get_root_xz_indices(representation)

    # We need one extra frame because x/z target at time t stores difference t -> t+1.
    total_len = initial_seq_len + eval_frames + 1

    full_seq = sample_batch(dances, batch_size, total_len, frame_rate)
    full_seq_dif = convert_root_xz_to_differences(full_seq, x_idx, z_idx)

    seed_seq_dif = full_seq_dif[:, :initial_seq_len - 1]
    target_future = full_seq_dif[:, initial_seq_len:initial_seq_len + eval_frames]

    seed_tensor = torch.autograd.Variable(torch.FloatTensor(seed_seq_dif).cuda())

    with torch.no_grad():
        pred_seq = model.forward(seed_tensor, eval_frames)

    pred_np = pred_seq.detach().cpu().numpy().reshape(batch_size, -1, full_seq.shape[2])

    # Model output includes predictions during the seed/warmup phase.
    pred_future = pred_np[:, initial_seq_len - 1:initial_seq_len - 1 + eval_frames]

    err = pred_future - target_future

    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    return mse, rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Evaluate first-step motion prediction error.")
    parser.add_argument("--representation", choices=["pos", "euler", "quad"], required=True)
    parser.add_argument("--dances_folder", required=True)
    parser.add_argument("--read_weight_path", required=True)
    parser.add_argument("--in_frame", type=int, required=True)
    parser.add_argument("--out_frame", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_seq_len", type=int, default=20)
    parser.add_argument("--eval_frames", type=int, default=20)
    parser.add_argument("--dance_frame_rate", type=int, default=60)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output_json", default="")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because the project model code uses .cuda().")

    dances = load_dances(args.dances_folder)

    model_class = get_model_class(args.representation)
    model = model_class(args.in_frame, args.hidden_size, args.out_frame)
    model.load_state_dict(torch.load(args.read_weight_path))
    model.cuda()
    model.eval()

    mses = []
    rmses = []
    maes = []

    for i in range(args.trials):
        mse, rmse, mae = evaluate_once(
            model=model,
            dances=dances,
            representation=args.representation,
            batch_size=args.batch_size,
            initial_seq_len=args.initial_seq_len,
            eval_frames=args.eval_frames,
            frame_rate=args.dance_frame_rate,
        )
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        print(f"trial {i:03d}: mse={mse:.6f}, rmse={rmse:.6f}, mae={mae:.6f}")

    result = {
        "representation": args.representation,
        "dances_folder": args.dances_folder,
        "read_weight_path": args.read_weight_path,
        "in_frame": args.in_frame,
        "out_frame": args.out_frame,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "initial_seq_len": args.initial_seq_len,
        "eval_frames": args.eval_frames,
        "trials": args.trials,
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
    }

    print("---- summary ----")
    print(json.dumps(result, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
