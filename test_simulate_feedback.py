import torch
import torch.nn as nn
import pickle
import argparse
import math
import numpy as np
from train_TC import LSTMNet
from train_TC import FFNet
from matplotlib import pyplot as plt
import os
import time
from torch.autograd import grad
def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

def load_data(data_path, seg, sample_rate=100):
    data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    time = data[:, 0]
    tendon_disp = data[:, 1]
    tip_A = data[:, 8]
    # resample data using fixed sampling rate
    # sample_rate = 100  # Hz
    frequency = round(1 / (time[-1] / 7), 2)
    interp_time = np.arange(time[0], time[-1], 1/sample_rate)
    tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
    tip_A_resample = np.interp(interp_time, time, tip_A)

    # normalization to [0, 1] and pad -1
    tendon_disp = np.hstack([-np.ones(seg), tendon_disp_resample/6])
    tip_A = np.hstack([-np.ones(seg), tip_A_resample/90])
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'freq': freq}

if __name__ == '__main__':
    forward_path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch488_best0.00010981084632440833.pt"
    inverse_path = "./checkpoints/Train12345Hz1rep_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch477_best0.0001874904317139533.pt"

    forward = False
    pos1 = 0
    sr = 25
    fq, input_dim = False, 1
    act = None
    # load trained model
    device = "cpu"
    model = LSTMNet(inp_dim=input_dim, num_layers=2, act=act)
    model.load_state_dict(torch.load(forward_path, map_location=device))
    # model.cuda()
    model.eval()

    inv_model = LSTMNet(inp_dim=input_dim, num_layers=2, act=act)
    inv_model.load_state_dict(torch.load(inverse_path, map_location=device))
    # model.cuda()
    inv_model.eval()

    seg = 50
    # load data
    path = "./tendon_data/45ca_1rep/test/0BL_015Hz_repn.txt"
    # path = "./tendon_data/45ca_1rep/random_withpause5.txt"
    data_path = os.path.join(path)
    data = load_data(data_path, seg, sample_rate=sr)

    tendons = data['tendon_disp'][:, np.newaxis].astype("float32")
    angles = data['tip_A'][:, np.newaxis].astype("float32")

    out = []
    out_feedback = []
    pre_angles = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    forward_h_ = hidden
    feedback_h_ = hidden
    _ = hidden
    torch.cuda.synchronize()
    start_time = time.time()
    t = 0
    for i in range(data['tendon_disp'].shape[0]):
        joint = angles[i:i + 1, 0:input_dim]
        input_device = torch.tensor(np.array([joint])).to(device)
        output, h_ = inv_model(input_device, h_)
        pre_tendon = output.detach().cpu().numpy()[0]
        out.append(pre_tendon[0])

        # predict tendon disp. with designated tip angle
        if i > 640:
            feedback_out, _ = inv_model(input_device, feedback_h_)
        else:
            feedback_out, _ = inv_model(input_device, _)
        pre_tendon_feedback = feedback_out.detach().cpu().numpy()[0]
        out_feedback.append(pre_tendon_feedback[0])
        # use forward kinematics model to simulate tip angles
        tendon = pre_tendon_feedback
        tendon_device = torch.tensor(np.array([tendon])).to(device)
        forward_out, forward_h_ = model(tendon_device, forward_h_)
        pre_angle = forward_out
        pre_angles.append(pre_angle.detach().cpu().numpy()[0][0])

        o_, feedback_h_ = inv_model(pre_angle, feedback_h_)

    out = np.array(out)
    out_feedback = np.array(out_feedback)
    pre_angles = np.array(pre_angles)
    print(out.shape, out_feedback.shape)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4*1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(data['tip_A'][seg:] * 90, data['tendon_disp'][seg:] * 6, linewidth=0.8, label="Ground truth")
    plt.plot(data['tip_A'][seg:] * 90, out[seg:, 0] * 6, linewidth=0.8, label="LSTM model")
    # plt.plot(data['tip_A'][seg:] * 90, out_feedback[seg:, 0] * 6, linewidth=0.8, label="Feedback")
    plt.ylim([-0.5, 6.5])
    plt.xlim([0, 90])
    plt.ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
    plt.xlabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
    plt.title("RMSE:{:.3f} (mm)".format(rmse_norm(out[seg:, 0] * 6, data['tendon_disp'][seg:] * 6)), fontsize=8)
    plt.legend(fontsize=8, frameon=False)

    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(data["time"], data['tendon_disp'][seg:] * 6, linewidth=0.8, label="Ground truth")
    plt.plot(data["time"], out[seg:, 0] * 6, linewidth=0.8, label="LSTM model")
    # plt.plot(data["time"], out_feedback[seg:, 0] * 6, linewidth=0.8, label="Feedback")
    # plt.plot(data["time"], pre_angles[seg:, 0] * 6, linewidth=0.8, label="predicted angle")
    plt.xlabel("Time (s)", fontsize=8, labelpad=0.01)
    plt.ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
    plt.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.{}Hz_pos{}.jpg".format(test_freq[0], pos1))
    plt.show()


