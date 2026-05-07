import os
from os import listdir

import numpy as np
import read_bvh
import transforms3d.euler as t3d_euler
import transforms3d.quaternions as t3d_quat


TRANSLATION_SCALE = 0.01

# BVH raw motion format here:
#   channels 0:3 = root translation
#   channels 3:  = Euler rotations in triplets
#
# In this dataset/Yiangos code, BVH rotation channels are in Z, X, Y order.
# We convert each (Z, X, Y) triplet to quaternion (w, x, y, z).
BVH_EULER_AXES = "rzxy"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_trailing_slash(path):
    return path if path.endswith("/") else path + "/"


def euler_triplet_zxy_degrees_to_quat(euler_zxy_deg):
    z_deg, x_deg, y_deg = euler_zxy_deg
    z, x, y = np.deg2rad([z_deg, x_deg, y_deg])
    q = t3d_euler.euler2quat(z, x, y, axes=BVH_EULER_AXES)
    return q


def quat_to_euler_triplet_zxy_degrees(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    z, x, y = t3d_euler.quat2euler(q, axes=BVH_EULER_AXES)
    return np.rad2deg([z, x, y])


def encode_quaternion_frame(raw_frame):
    raw_frame = np.asarray(raw_frame, dtype=float)

    translation = raw_frame[0:3].copy() * TRANSLATION_SCALE
    euler_values = raw_frame[3:]

    if len(euler_values) % 3 != 0:
        raise ValueError(f"Euler rotation channel count is not divisible by 3: {len(euler_values)}")

    quats = []
    for i in range(0, len(euler_values), 3):
        q = euler_triplet_zxy_degrees_to_quat(euler_values[i:i + 3])
        quats.extend(q)

    return np.concatenate([translation, np.array(quats, dtype=float)])


def decode_quaternion_frame(quad_frame):
    quad_frame = np.asarray(quad_frame, dtype=float)

    translation = quad_frame[0:3].copy() / TRANSLATION_SCALE
    quat_values = quad_frame[3:]

    if len(quat_values) % 4 != 0:
        raise ValueError(f"Quaternion channel count is not divisible by 4: {len(quat_values)}")

    eulers = []
    for i in range(0, len(quat_values), 4):
        euler_triplet = quat_to_euler_triplet_zxy_degrees(quat_values[i:i + 4])
        eulers.extend(euler_triplet)

    return np.concatenate([translation, np.array(eulers, dtype=float)])


def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    src_bvh_folder = ensure_trailing_slash(src_bvh_folder)
    tar_traindata_folder = ensure_trailing_slash(tar_traindata_folder)
    ensure_dir(tar_traindata_folder)

    bvh_dance_names = sorted(listdir(src_bvh_folder))

    for bvh_dance_name in bvh_dance_names:
        if not bvh_dance_name.endswith(".bvh"):
            continue

        src_path = src_bvh_folder + bvh_dance_name
        raw_motion = read_bvh.parse_frames(src_path)

        quad_motion = np.array([encode_quaternion_frame(frame) for frame in raw_motion])
        np.save(tar_traindata_folder + bvh_dance_name + ".npy", quad_motion)

        print(f"encoded {src_path} -> {tar_traindata_folder + bvh_dance_name + '.npy'} {quad_motion.shape}")


def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder):
    src_train_folder = ensure_trailing_slash(src_train_folder)
    tar_bvh_folder = ensure_trailing_slash(tar_bvh_folder)
    ensure_dir(tar_bvh_folder)

    dance_names = sorted(listdir(src_train_folder))

    for dance_name in dance_names:
        if not dance_name.endswith(".npy"):
            continue

        src_path = src_train_folder + dance_name
        quad_motion = np.load(src_path)

        raw_motion = np.array([decode_quaternion_frame(frame) for frame in quad_motion])

        out_path = tar_bvh_folder + dance_name + ".bvh"
        read_bvh.write_frames("train_data_bvh/standard.bvh", out_path, raw_motion)

        print(f"decoded {src_path} -> {out_path} {raw_motion.shape}")


def main():
    bvh_dir_path = "train_data_bvh/martial/"
    quad_enc_dir_path = "train_data_quad/martial/"
    bvh_reconstructed_dir_path = "reconstructed_bvh_data_quad/martial/"

    generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)
    generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)


if __name__ == "__main__":
    main()
