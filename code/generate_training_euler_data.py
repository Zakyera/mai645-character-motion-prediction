import os
from os import listdir

import numpy as np
import read_bvh


TRANSLATION_SCALE = 0.01


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_trailing_slash(path):
    return path if path.endswith("/") else path + "/"


def encode_euler_frame(raw_frame):
    """BVH raw frame -> Euler training frame.

    The BVH frame already contains:
    - root/hip translation in channels 0:3
    - Euler rotation channels after that

    We scale only root translation so its magnitude is closer to the rotation values.
    """
    frame = raw_frame.copy()
    frame[0:3] *= TRANSLATION_SCALE
    return frame


def decode_euler_frame(train_frame):
    """Euler training frame -> BVH raw frame."""
    frame = train_frame.copy()
    frame[0:3] /= TRANSLATION_SCALE
    return frame


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    src_bvh_folder = ensure_trailing_slash(src_bvh_folder)
    tar_traindata_folder = ensure_trailing_slash(tar_traindata_folder)
    ensure_dir(tar_traindata_folder)

    bvh_dance_names = sorted(listdir(src_bvh_folder))

    for bvh_dance_name in bvh_dance_names:
        if not bvh_dance_name.endswith(".bvh"):
            continue

        src_path = src_bvh_folder + bvh_dance_name
        raw_motion = read_bvh.parse_frames(src_path)

        train_motion = np.array([encode_euler_frame(frame) for frame in raw_motion])
        np.save(tar_traindata_folder + bvh_dance_name + ".npy", train_motion)

        print(f"encoded {src_path} -> {tar_traindata_folder + bvh_dance_name + '.npy'} {train_motion.shape}")


def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    src_train_folder = ensure_trailing_slash(src_train_folder)
    tar_bvh_folder = ensure_trailing_slash(tar_bvh_folder)
    ensure_dir(tar_bvh_folder)

    dance_names = sorted(listdir(src_train_folder))

    for dance_name in dance_names:
        if not dance_name.endswith(".npy"):
            continue

        src_path = src_train_folder + dance_name
        train_motion = np.load(src_path)

        raw_motion = np.array([decode_euler_frame(frame) for frame in train_motion])

        out_path = tar_bvh_folder + dance_name + ".bvh"
        read_bvh.write_frames("train_data_bvh/standard.bvh", out_path, raw_motion)

        print(f"decoded {src_path} -> {out_path} {raw_motion.shape}")


def main():
    bvh_dir_path = "train_data_bvh/martial/"
    euler_enc_dir_path = "train_data_euler/martial/"
    bvh_reconstructed_dir_path = "reconstructed_bvh_data_euler/martial/"

    generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)
    generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)


if __name__ == "__main__":
    main()
