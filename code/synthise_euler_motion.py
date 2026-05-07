import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

import read_bvh


In_frame_size = 132
TRANSLATION_SCALE = 0.01

# In raw BVH/Euler representation, root/hip translation is channels 0, 1, 2.
HIP_X_INDEX = 0
HIP_Z_INDEX = 2


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=132, hidden_size=1024, out_frame_size=132):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    def init_hidden(self, batch):
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())

        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())

        return [h0, h1, h2], [c0, c1, c2]

    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h0, (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h1, (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)

        return out_frame, [vec_h0, vec_h1, vec_h2], [vec_c0, vec_c1, vec_c2]

    def forward(self, initial_seq, generate_frames_number):
        batch = initial_seq.size()[0]

        vec_h, vec_c = self.init_hidden(batch)

        out_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, 1))).cuda())
        out_frame = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.out_frame_size))).cuda())

        for i in range(initial_seq.size()[1]):
            in_frame = initial_seq[:, i]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        for _ in range(generate_frames_number):
            in_frame = out_frame
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        return out_seq[:, 1: out_seq.size()[1]]


def ensure_trailing_slash(path):
    return path if path.endswith("/") else path + "/"


def load_dances(dance_folder):
    dance_folder = ensure_trailing_slash(dance_folder)
    dance_files = sorted(os.listdir(dance_folder))
    dances = []

    for dance_file in dance_files:
        if not dance_file.endswith(".npy"):
            continue

        print("load " + dance_file)
        dance = np.load(dance_folder + dance_file)
        print("frame number: " + str(dance.shape[0]))
        dances.append(dance)

    return dances


def get_dance_len_lst(dances):
    len_lst = []
    for _ in dances:
        len_lst.append(10)

    index_lst = []
    for index, length in enumerate(len_lst):
        for _ in range(length):
            index_lst.append(index)

    return index_lst


def write_euler_traindata_to_bvh(bvh_filename, euler_train_data):
    raw_motion = np.array(euler_train_data).copy()

    # Undo translation scaling from preprocessing.
    raw_motion[:, 0:3] = raw_motion[:, 0:3] / TRANSLATION_SCALE

    read_bvh.write_frames("train_data_bvh/standard.bvh", bvh_filename, raw_motion)


def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):
    # Match the training script: root x/z are learned as frame-to-frame differences.
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0:initial_seq_np.shape[1] - 1]
    initial_seq_dif_np = initial_seq_np[:, 0:initial_seq_np.shape[1] - 1].copy()

    initial_seq_dif_np[:, :, HIP_X_INDEX] = dif[:, :, HIP_X_INDEX]
    initial_seq_dif_np[:, :, HIP_Z_INDEX] = dif[:, :, HIP_Z_INDEX]

    initial_seq = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_np.tolist()).cuda())

    predict_seq = model.forward(initial_seq, generate_frames_number)

    batch = initial_seq_np.shape[0]

    for b in range(batch):
        out_seq = np.array(predict_seq[b].data.tolist()).reshape(-1, In_frame_size)

        # Convert predicted root x/z differences back to cumulative trajectory.
        last_x = 0.0
        last_z = 0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame, HIP_X_INDEX] = out_seq[frame, HIP_X_INDEX] + last_x
            last_x = out_seq[frame, HIP_X_INDEX]

            out_seq[frame, HIP_Z_INDEX] = out_seq[frame, HIP_Z_INDEX] + last_z
            last_z = out_seq[frame, HIP_Z_INDEX]

        write_euler_traindata_to_bvh(save_dance_folder + "out" + "%02d" % b + ".bvh", out_seq)

    return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)


def test(dances, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
         write_bvh_motion_folder, in_frame_size=132, hidden_size=1024, out_frame_size=132):

    torch.cuda.set_device(0)

    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
    model.load_state_dict(torch.load(read_weight_path))
    model.cuda()
    model.eval()

    dance_len_lst = get_dance_len_lst(dances)
    random_range = len(dance_len_lst)

    speed = frame_rate / 30

    dance_batch = []

    for _ in range(batch):
        dance_id = dance_len_lst[np.random.randint(0, random_range)]
        dance = dances[dance_id].copy()
        dance_len = dance.shape[0]

        start_id = random.randint(10, int(dance_len - initial_seq_len * speed - 10))

        sample_seq = []
        for i in range(initial_seq_len):
            sample_seq.append(dance[int(i * speed + start_id)])

        dance_batch.append(sample_seq)

    dance_batch_np = np.array(dance_batch)

    with torch.no_grad():
        generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)


def main():
    parser = argparse.ArgumentParser(description="Synthesize Euler motion using a trained acLSTM model.")

    parser.add_argument("--dances_folder", type=str, required=True)
    parser.add_argument("--read_weight_path", type=str, required=True)
    parser.add_argument("--write_bvh_motion_folder", type=str, required=True)
    parser.add_argument("--dance_frame_rate", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--initial_seq_len", type=int, default=15)
    parser.add_argument("--generate_frames_number", type=int, default=400)
    parser.add_argument("--in_frame", type=int, default=132)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--out_frame", type=int, default=132)

    args = parser.parse_args()

    dances_folder = ensure_trailing_slash(args.dances_folder)
    write_bvh_motion_folder = ensure_trailing_slash(args.write_bvh_motion_folder)

    os.makedirs(write_bvh_motion_folder, exist_ok=True)

    dances = load_dances(dances_folder)

    test(
        dances,
        args.dance_frame_rate,
        args.batch_size,
        args.initial_seq_len,
        args.generate_frames_number,
        args.read_weight_path,
        write_bvh_motion_folder,
        args.in_frame,
        args.hidden_size,
        args.out_frame,
    )


if __name__ == "__main__":
    main()
