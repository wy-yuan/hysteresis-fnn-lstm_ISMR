import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import torch
import sys
sys.path.append("./")
from models.hybrid_model import BacklashNet
from models.hybrid_model import BacklashInverseNet
from train_TC import LSTMNet
import torch.nn as nn
import scipy.stats as stats
import seaborn as sns

def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

def load_data(data_path, seg, sample_rate=100):
    data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    time = data[:, 0]
    tendon_disp = data[:, 1]
    tip_A = data[:, 8]
    current = data[:, 3]
    em_pos = data[:, 5:8]
    # tip_disp = np.linalg.norm(em_pos, axis=1)
    tip_disp = data[:, 5]
    X = 7.98 * 25.4 - data[:, 5] + 14  # unit mm
    Y = data[:, 6] - 1.13 * 25.4  # unit mm
    # resample data using fixed sampling rate
    # sample_rate = 100  # Hz
    frequency = round(1 / (time[-1] / 7), 2)
    interp_time = np.arange(10, time[-1], 1 / sample_rate)
    tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
    tip_A_resample = np.interp(interp_time, time, tip_A)
    current_resample = np.interp(interp_time, time, current)
    tip_disp_resample = np.vstack([np.interp(interp_time, time, X), np.interp(interp_time, time, Y)]).T

    # normalization to [0, 1] and pad -1
    tendon_disp = np.hstack([np.ones(seg)*0, tendon_disp_resample / 6]) #6
    tip_A = np.hstack([np.ones(seg)*30/90, tip_A_resample / 90]) #90
    tip_D = np.vstack([np.hstack([np.ones((seg,1))*66/90, np.ones((seg,1))*20/90]), tip_disp_resample / 90])
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'tip_D': tip_D, 'freq': freq, 'current': current_resample}

def test_backlash(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, tip_name="tip_A"):
    device = "cuda"
    if forward:
        model = BacklashNet()
    else:
        model = BacklashInverseNet()
    sr = sr
    seg = 0
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    # tip_name = "tip_D"
    data = load_data(data_path, seg, sample_rate=sr)
    # data[tip_name][:, 1] = data_filter(data[tip_name][:,1], datatype="disp")  # It's for inverse kinematic modeling

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    if tip_name == "tip_D":
        pre_pos = data[tip_name][0:1, 1:2].astype("float32")  # for tipY only
        tips = data[tip_name][:, 1:2].astype("float32")  # for tipY only
    else:
        pre_pos = data[tip_name][0:1, np.newaxis].astype("float32")
        tips = data[tip_name][:, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    out = []
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        tip = tips[i:i + 1, 0:input_dim]
        if forward:
            output, dynamic_weights, condition_lo, condition_up, condition_mid = model(torch.tensor([joint]).to(device), torch.tensor([pre_pos]).to(device))
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        else:
            output, dynamic_weights = model(torch.tensor([tip]).to(device), torch.tensor([pre_pos]).to(device), torch.tensor([pre_td]).to(device))  # x_curr, _x_prev, u_pre
            pre_td = output.detach().cpu().numpy()[0]
            out.append(pre_td[0])
            pre_pos = tip
    print(dynamic_weights)
    out = np.array(out)
    if forward:
        input = data['tendon_disp'][seg:] * 6
        if tip_name == "tip_D":
            gt = data[tip_name][seg:, 1] * 90
        else:
            gt = data[tip_name][seg:] * 90
        output = out[seg:, 0] * 90
    # else:
    #     input = data[tip_name][seg:, 1] * 90
    #     gt = data['tendon_disp'][seg:] * 6
    #     output = out[seg:, 0] * 6
    return input, gt, output, data['time']

def test_LSTM(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    model = LSTMNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()
    tip_name = 'tip_A'
    data = load_data(data_path, seg, sample_rate=sr)
    # data[tip_name] = data_filter(data[tip_name])

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data[tip_name][:, np.newaxis].astype("float32")
    if fq:
        freq = data['freq'][:, np.newaxis].astype("float32")
        joints = np.concatenate([joints, freq], axis=1)

    out = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)
    if forward:
        input = data['tendon_disp'][seg:]*6
        gt = data[tip_name][seg:]*90
        output = out[seg:, 0]*90
    else:
        input = data[tip_name][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    gtx = data["tip_D"][seg:, 0] * 90
    gty = data["tip_D"][seg:, 1] * 90
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))
    return input, gt, gtx, gty, output, data['time'], res_rmse, res_nrmse

class fixedBacklash(nn.Module):
    def __init__(self):
        super(fixedBacklash, self).__init__()

    def h(self, s):
        s_copy = s.clone()
        s_copy[s > 0] = 0
        s_copy[s <= 0] = 1
        return s_copy

    def forward(self, x, _out_prev):
        m_lo, m_up, c_lo, c_up = torch.tensor(2), torch.tensor(1.8), torch.tensor(0.5), torch.tensor(-0.5) #0.6847, 0.7132, 0.4703, 0.3338
        f1 = self.h((m_lo.detach() * x + m_lo.detach() * c_lo.detach() - _out_prev) / m_lo.detach())
        f2 = self.h((_out_prev - m_up.detach() * x - m_up.detach() * c_up.detach()) / m_up.detach())
        out_curr = m_lo * x * f1 + m_lo * c_lo * f1 + m_up * x * f2 + m_up * c_up * f2 + _out_prev * (1 - f1) * (1 - f2)
        return out_curr

class GRDPINet(nn.Module):
    def __init__(self):
        super(GRDPINet, self).__init__()
        self.weights = nn.Parameter(torch.ones(11)*0.5)
        # self.weights = nn.Parameter(torch.tensor([0.5, 0.35, 0.25, 0.2, 0.15, 2, 0.5, 0.28, 50, 1, 0.01]))
        # self.weights = nn.Parameter(torch.randn(11))
        self.tanh = nn.Tanh()

    def zi(self, u, r, z_pre):
        # z = torch.max(u-r, torch.min(u+r, z_pre))
        z = (u - r) + torch.relu(((u + r) - torch.relu((u + r) - z_pre)) - (u - r))
        # leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        return z

    def ri(self, h, i, beta, du):
        # return h*i + beta*torch.abs(du)
        return h*i + beta*torch.sqrt(du*du + 1e-10)

    def OneStep_forward(self, u, u_pre, z_pre):
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
        #                                               , self.weights[5], self.weights[6], self.weights[7], self.weights[8], self.weights[9], self.weights[10])
        a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 2, 0, 0, 0, 0
        du = (u - u_pre)
        r1 = self.ri(h, 1, beta, du)
        r2 = self.ri(h, 2, beta, du)
        r3 = self.ri(h, 3, beta, du)
        r4 = self.ri(h, 4, beta, du)
        z1, z2 = self.zi(u, r1, z_pre[:, :, 0:1]), self.zi(u, r2, z_pre[:, :, 1:2])
        z3, z4 = self.zi(u, r3, z_pre[:, :, 2:3]), self.zi(u, r4, z_pre[:, :, 3:4])
        out_curr = a0*u + a1*z1 + a2*z2 + a3*z3 + a4*z4 + b
        z = torch.cat([z1, z2, z3, z4], dim=2)
        # out_curr = c3 * out_curr + b
        return out_curr, z

    def forward(self, x, x_pre, z_init):
        z = z_init  # shape: (batch_size, 1, 4)
        custom_output = torch.empty(x.shape[0], 0, 1).to("cuda")
        for i in range(x.shape[1]):
            out_i, z = self.OneStep_forward(x[:, i:i + 1, :], x_pre[:, i:i + 1, :], z)
            custom_output = torch.cat((custom_output, out_i), dim=1)
        return custom_output, self.weights, z

def plot_figure3():
    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4))
    plt.rcParams['font.family'] = 'Times New Roman'

    # Backlash model
    fixed_backlash = fixedBacklash()
    fixed_backlash.eval()
    input_data = np.hstack([np.linspace(0, 1, 100), np.linspace(1, -1, 100), np.linspace(-1, 1, 100)])
    output_data = []
    out_bl = 0
    for i in input_data:
        out_bl = fixed_backlash(i, out_bl)
        output_data.append(out_bl)
    plt.subplot(2, 2, 1)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(input_data, output_data, 'b', linewidth=0.8)
    plt.xlabel("Input", fontsize=8, labelpad=0.3)
    plt.ylabel("Output", fontsize=8, labelpad=0.3)
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim([-1.2, 1.2])
    plt.grid(True, linestyle=':')

    # PI model
    model = GRDPINet()
    model.cuda()
    model.eval()
    out = []
    joints_ = torch.tensor(np.array(input_data)[:, np.newaxis, np.newaxis]).to("cuda")
    h_init = joints_[0:1, :, :].expand(1, 1, 4)
    h_ = h_init
    for i in range(len(input_data)):
        input_ = joints_[i:i + 1, 0:1]
        if i == 0:
            input_pre = joints_[i:i + 1, 0:1]
        else:
            input_pre = joints_[i - 1:i, 0:1]
        output, weights, h_ = model(input_, input_pre, h_)
        pre_pos = output.detach().cpu().numpy()[0]
        # print(pre_pos.shape)
        out.append(pre_pos[0])

    plt.subplot(2, 2, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(input_data, out, 'b', linewidth=0.8)
    plt.xlabel("Input", fontsize=8, labelpad=0.3)
    plt.ylabel("Output", fontsize=8, labelpad=0.3)
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim([-1.2, 1.2])
    plt.grid(True, linestyle=':')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.1, bottom=0.15, left=0.1, right=0.98)
    plt.savefig("./figures/RAL_plots/Fig3.png", dpi=600)
    plt.savefig("./figures/RAL_plots/Fig3.svg")
    plt.show()


def plot_figure5():
    plt.figure(figsize=(88.9 / 25.4*2, 88.9 / 25.4/1.8))
    plt.rcParams['font.family'] = 'Times New Roman'

    folder_path = r"D:\yuan\Dropbox (BCH)\bch\IRAL\figures\ISMR2024Datasets_0.5Hz"
    i = 0
    for file in os.listdir(folder_path):
        i += 1
        data_path = os.path.join(folder_path, file)
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        tendon_disp = data[:, 1]
        tip_A = data[:, 8]
        current = data[:, 3]
        force = current/100/0.6*6.56    # current is in cA
        plt.subplot(1, 3, 3)
        plt.tick_params(labelsize=8, pad=0.01, length=2)
        plt.plot(tendon_disp, tip_A, linewidth=0.8, label="Trial{}".format(i))
        plt.xlabel("Tendon displacement (mm)", fontsize=8, labelpad=0.5)
        plt.ylabel("Tip angle (degrees)", fontsize=8, labelpad=0.5)
        plt.legend(fontsize=7, frameon=False, ncol=1, handlelength=1)
        plt.xlim([-0.2, 6.2])
        plt.xticks([0, 1, 2, 3, 4, 5, 6])
        plt.ylim([15, 80])

        # plt.subplot(1, 3, 3)
        # plt.tick_params(labelsize=8, pad=0.01, length=2)
        # plt.plot(force, tip_A, linewidth=0.8, label="Trial{}".format(i))
        # plt.xlabel("Tendon tension (N)", fontsize=8, labelpad=0.5)
        # plt.ylabel("Tip angle (degree)", fontsize=8, labelpad=0.5)
        # plt.xlim([-0.2, 2.7])
        # plt.xticks([0, 0.5, 1, 1.5, 2, 2.5])
        # plt.ylim([15, 80])
        # plt.legend(fontsize=7, frameon=False, ncol=1, handlelength=1)

    folder_path = r"D:\yuan\Dropbox (BCH)\bch\IRAL\figures\0_baseline"   # show rate dependence
    i = 0
    for file in os.listdir(folder_path):
        i += 1
        data_path = os.path.join(folder_path, file)
        # data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        # tendon_disp = data[:, 1]
        # tip_A = data[:, 8]
        data = load_data(data_path, 0, sample_rate=25)
        tendon_disp = data["tendon_disp"] * 6
        tip_A = data["tip_A"] * 90
        plt.subplot(1, 3, 1)
        plt.tick_params(labelsize=8, pad=0.01, length=2)
        plt.plot(tendon_disp, tip_A, linewidth=0.8, label="0.{} Hz".format(i))
        plt.xlabel("Tendon displacement (mm)", fontsize=8, labelpad=0.5)
        plt.ylabel("Tip angle (degrees)", fontsize=8, labelpad=0.5)
        plt.legend(fontsize=7, frameon=False, ncol=1, handlelength=1)
        plt.xlim([-0.2, 6.2])
        plt.xticks([0, 1, 2, 3, 4, 5, 6])
        plt.ylim([15, 80])

        current = data["current"]
        force = current / 100 / 0.6 * 6.56 # current is in cA
        plt.subplot(1, 3, 2)
        plt.tick_params(labelsize=8, pad=0.01, length=2)
        plt.plot(force, tip_A, linewidth=0.8, label="0.{} Hz".format(i))
        plt.xlabel("Tendon tension (N)", fontsize=8, labelpad=0.5)
        plt.ylabel("Tip angle (degrees)", fontsize=8, labelpad=0.5)
        plt.legend(fontsize=7, frameon=False, ncol=1, handlelength=1)
        plt.xlim([-2, 27])
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.ylim([15, 80])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.1, bottom=0.15, left=0.05, right=0.98)
    plt.savefig("./figures/RAL_plots/Fig5.png", dpi=600)
    plt.savefig("./figures/RAL_plots/Fig5.svg")
    plt.show()

def plot_figure6():
    # data_path = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus/0BL_0.45Hz.txt"
    data_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/random_stair1-10_20.txt"
    data_name = os.path.basename(data_path).split(".txt")[0]
    coef_type = "all"
    coef_x = [-3.62136415e-03, -4.61111115e-02, 7.04893736e+01]
    coef_y = [-0.00326109, 0.77266619, 0.21515682]

    # fit parameters using sinusoidal data
    # coef_type = "sinus"
    # coef_x = [-3.38357987e-03, -6.72903484e-02, 7.05323127e+01]
    # coef_y = [-0.00313098, 0.75053007, 0.58506313]
    p_x = np.poly1d(coef_x)
    p_y = np.poly1d(coef_y)

    # Backlash model one-stage mapping for Y
    model_path = "./checkpoints/TipY_Sinus12_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch100_best5.017787543692975e-05.pt"
    input_, gt, output, time = test_backlash(model_path, data_path, seg=50, forward=True, input_dim=1, tip_name="tip_D")

    # Backlash model for angle prediction
    model_path_angle = "./checkpoints/Sinus12_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch43_best7.098719712909467e-05.pt"
    input_, gt_A, output_A, time = test_backlash(model_path_angle, data_path, seg=50, forward=True, input_dim=1, tip_name="tip_A")
    two_stage_output = p_y(output_A)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4))
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.plot(input_, p_y(gt_A))
    # plt.plot(input_, p_y(output_A))
    # plt.plot(input_, gt)
    plt.subplot(2, 2, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(input_, gt, "b", linewidth=0.8, label="Experimental data")
    plt.plot(input_, output, "g", linewidth=0.8, label="Direct map")
    plt.plot(input_, two_stage_output, "r", linewidth=0.8, label="Two-part maps")
    plt.xlabel("Tendon displacement (mm)", fontsize=8, labelpad=0.5)
    plt.ylabel("Tip position Y (mm)", fontsize=8, labelpad=0.5)
    plt.legend(fontsize=7, frameon=False, ncol=1, handlelength=1)
    plt.xlim([-0.2, 6.2])
    plt.ylim([19, 46])
    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(time-10, gt, "b", linewidth=0.8, label="Experimental data")
    plt.plot(time-10, output, "g", linewidth=0.8, label="Direct map")
    plt.plot(time-10, two_stage_output, "r", linewidth=0.8, label="Two-part maps")
    plt.xlabel("Time (s)", fontsize=8, labelpad=0.5)
    plt.ylabel("Tip position Y (mm)", fontsize=8, labelpad=0.5)
    plt.xlim([-1, 81])
    plt.ylim([19, 46])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.18, bottom=0.08, left=0.1, right=0.98)
    plt.legend(fontsize=7, frameon=False, ncol=1, handlelength=1)
    plt.savefig("./figures/RAL_plots/Fig6_{}_{}.png".format(coef_type, data_name), dpi=600)
    plt.savefig("./figures/RAL_plots/Fig6_{}_{}.svg".format(coef_type, data_name))
    plt.show()

def plot_figure7():
    sinus_data_path = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus/MidBL_0.45Hz.txt"
    sinus_data_name = os.path.basename(sinus_data_path).split(".txt")[0]
    stop_data_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/random_stair1-10_11.txt"
    stop_data_name = os.path.basename(stop_data_path).split(".txt")[0]

    sinus_model_path = "./checkpoints/Sinus12_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch88_best6.226386176422238e-05.pt"
    sinus_model_STDloss_path = "./checkpoints/Sinus12_LSTM_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch63_best0.00011929163732323407.pt"
    stop_model_path = "./checkpoints/Stair5s_LSTM_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch89_best0.00045744293472833105.pt"
    input_s, gt_s, gtx, gty, output_s_s, time_s, res_rmse1, res_nrmse1 = test_LSTM(sinus_model_path, sinus_data_path, seg=0, forward=True, input_dim=1)
    # input_st, gt_st, gtx, gty, output_s_st, time_st, res_rmse1, res_nrmse1 = test_LSTM(sinus_model_path, stop_data_path, seg=0, forward=True, input_dim=1)
    input_st, gt_st, gtx, gty, output_s_st, time_st, res_rmse1, res_nrmse1 = test_LSTM(sinus_model_STDloss_path, stop_data_path, seg=0, forward=True, input_dim=1)
    input_s, gt_s, gtx, gty, output_st_s, time_s, res_rmse1, res_nrmse1 = test_LSTM(stop_model_path, sinus_data_path, seg=0, forward=True, input_dim=1)
    input_st, gt_st, gtx, gty, output_st_st, time_st, res_rmse1, res_nrmse1 = test_LSTM(stop_model_path, stop_data_path, seg=0, forward=True, input_dim=1)

    colors = ["blue", "red", "green"]
    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(time_s-10, gt_s, colors[0], linewidth=0.8, label="Experimental data")
    plt.plot(time_s-10, output_s_s, colors[1], linewidth=0.8, label="Sinusoidal model")
    plt.plot(time_s-10, output_st_s, colors[2], linewidth=0.8, label="Point-to-point model")
    plt.xlabel("t (s)", fontsize=8, labelpad=0.5)
    plt.ylabel("θ (°)", fontsize=8, labelpad=0.5)
    plt.legend(fontsize=8, frameon=False, ncol=1, handlelength=1)
    plt.ylim([28, 92])
    plt.yticks([30, 40, 50, 60, 70, 80, 90])
    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(time_st-10, gt_st, colors[0], linewidth=0.8, label="Experimental data")
    plt.plot(time_st-10, output_s_st, colors[1], linewidth=0.8, label="Sinusoidal model")
    plt.plot(time_st-10, output_st_st, colors[2], linewidth=0.8, label="Point-to-point model")
    plt.xlabel("t (s)", fontsize=8, labelpad=0.5)
    plt.ylabel("θ (°)", fontsize=8, labelpad=0.5)
    plt.ylim([28, 92])
    plt.yticks([30, 40, 50, 60, 70, 80, 90])
    plt.legend(fontsize=8, frameon=False, ncol=1, handlelength=1)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.18, bottom=0.08, left=0.1, right=0.98)
    plt.savefig("./figures/RAL_plots/Fig7_bestloss.png".format(), dpi=600)
    plt.savefig("./figures/RAL_plots/Fig7_bestloss.svg".format())
    plt.show()

def plot_theta_error_merged_bestLoss8(version, model_name_list, str_list):
    plt.figure(figsize=(88.9 / 25.4*2, 88.9 / 25.4 / 1.3))
    plt.rcParams['font.family'] = 'Times New Roman'
    FS_FP_IS_IP = np.empty((16, 16), dtype='<U10')
    for number, model_type in enumerate(["forward", "inverse"]):
        unit = "degrees" if model_type == "forward" else "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        old_new_loss = []
        rmse_array_list = []  # data for paired t-test
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                model_list = [model_type + str + train_data for str in str_list]  #
                for test_data in ["sinus", "stair1_10s"]:
                    if train_data=="Sinus12" and test_data=="stair1_10s":
                        continue
                    if train_data=="Stair5s" and test_data=="sinus":
                        continue
                    for model_name in model_list:
                        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name, test_data))
                        rmse_me_StdL = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}StdL/{}_error.npy".format(model_name, test_data))
                        if np.mean(rmse_me[:, 0]) <= np.mean(rmse_me_StdL[:, 0]):
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                            me_dfs.append(np.max(rmse_me[:, 1]))
                            old_new_loss.append(0)
                            rmse_array_list.append(rmse_me[:, 0])
                        else:
                            rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me_StdL[:, 0]))
                            me_dfs.append(np.max(rmse_me_StdL[:, 1]))
                            old_new_loss.append(1)
                            rmse_array_list.append(rmse_me_StdL[:, 0])

        print(rmse_mean_dfs)
        print(me_dfs)
        tick_list = [i + 1 for i in range(len(model_name_list)*2)]
        plt.subplot(1, 2, number+1)
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        symbol_list = ["FNN", "BK", "PI", "LSTM", "GRU", "FNN-HIB", "HB1", "HB2"]
        symbol_list2 = ["FNN", "Backlash", "PI", "LSTM", "GRU", "FNN-HIB", "HB1: Serial LSTM-Backlash", "HB2: Parallel LSTM-Backlash"]

        # do the paired t-test and show the result in a plot
        print(model_type)

        m = []
        for i in range(8):
            n = []
            rmse_array1 = rmse_array_list[i]
            for j in range(8):
                if i == j:
                    n.append(np.nan)
                    if model_type == "forward":
                        FS_FP_IS_IP[i * 2, j * 2] = "_"
                    else:
                        FS_FP_IS_IP[i * 2, j * 2+1] = "_"
                    continue
                rmse_array2 = rmse_array_list[j]
                # t_stat, p_value = stats.ttest_rel(rmse_array1, rmse_array2)
                statistic, p_value = stats.wilcoxon(rmse_array1, rmse_array2)
                # n.append(round(p_value, 4))
                # n.append(p_value)
                n.append(f'{p_value:.{3-1}e}')
                if model_type == "forward":
                    if p_value < 0.05:
                        FS_FP_IS_IP[i*2, j*2] = "0"
                    else:
                        FS_FP_IS_IP[i*2, j*2] = "1"
                else:
                    if p_value < 0.05:
                        FS_FP_IS_IP[i*2, j*2+1] = "0"
                    else:
                        FS_FP_IS_IP[i*2, j*2+1] = "1"
                # print(f"Paired T-Test: T-statistic: {t_stat}, P-value: {p_value}")
                if j > i:
                    diff = rmse_array1 - rmse_array2
                    f = plt.figure(figsize=(10, 6))
                    sns.violinplot(diff)
                    sns.swarmplot(diff)
                    plt.savefig("./figures/RAL_plots/difference_dist/sinusoidal_{}_{}_{}.png".format(model_type, symbol_list[i], symbol_list[j]),
                                dpi=600)
                    plt.close(f)
            m.append(n)

        df = pd.DataFrame(m, columns=symbol_list, index=symbol_list)
        df.to_csv("./figures/RAL_plots/WilcoxonSignedRank/{}_sinusoidal.csv".format(model_type))

        m = []
        for i in range(8, 16):
            n = []
            rmse_array1 = rmse_array_list[i]
            for j in range(8, 16):
                if i == j:
                    n.append(np.nan)
                    if model_type == "forward":
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = "_"
                    else:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "_"
                    continue
                rmse_array2 = rmse_array_list[j]
                # t_stat, p_value = stats.ttest_rel(rmse_array1, rmse_array2)
                statistic, p_value = stats.wilcoxon(rmse_array1, rmse_array2)
                # print(f"Paired T-Test: T-statistic: {t_stat}, P-value: {p_value}")
                # n.append(round(p_value, 4))
                # n.append(p_value)
                n.append(f'{p_value:.{3-1}e}')
                if model_type == "forward":
                    if p_value < 0.05:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = "0"
                    else:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = "1"
                else:
                    if p_value < 0.05:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "0"
                    else:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "1"

                if j > i:
                    diff = rmse_array1 - rmse_array2
                    f = plt.figure(figsize=(10, 6))
                    sns.violinplot(diff)
                    sns.swarmplot(diff)
                    plt.savefig("./figures/RAL_plots/difference_dist/point_to_point_{}_{}_{}.png".format(model_type, symbol_list[i-8], symbol_list[j-8]),
                                dpi=600)
                    plt.close(f)
            m.append(n)
        df = pd.DataFrame(m, columns=symbol_list, index=symbol_list)
        df.to_csv("./figures/RAL_plots/WilcoxonSignedRank/{}_point_to_point.csv".format(model_type))

        # show the maximum value and the model name
        if model_type == "forward":
            offset = 0.12
        else:
            offset = 0.012
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            if old_new_loss[i] == 1:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i]-offset, symbol_list[i%8], ha='center', va='bottom', fontsize=6) #, fontweight='bold')
            else:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 8], ha='center', va='bottom', fontsize=6)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        # plt the error bar
        if version == "v1":
            # fmt_list = ['b'+f for f in ['o', '^', 'p', 'h', '*', '+', 'x']]
            color_list = ['brown', 'red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 8):
                for j in range(0, 2):
                    if j == 0:
                        plt.errorbar(tick_list[i+8*j], rmse_mean_dfs[i+8*j], yerr=rmse_std_dfs[i+8*j], fmt="o", color=color_list[i], label=symbol_list2[i], capsize=5) #color=color_list[i],
                    else:
                        plt.errorbar(tick_list[i+8*j], rmse_mean_dfs[i+8*j], yerr=rmse_std_dfs[i+8*j], fmt="o", color=color_list[i], capsize=5) #color=color_list[i],
            for x in np.arange(1, 2)*8+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 2) * 8 + 4, ["Sinusoids", "Point-to-point"])

        # if model_type == "forward":
        #     plt.ylim([0, 3.2])
        # else:
        #     plt.ylim([0, 0.37])
        # plt.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.ylabel("RMS Error / Maximum Error ({})".format(unit))
    plt.tight_layout()
    # plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/BestLoss_2groups_{}_{}.png".format(model_type, version), dpi=600)
    plt.savefig("./figures/RAL_plots/Fig8ab_shrink.png".format(), dpi=600)
    plt.savefig("./figures/RAL_plots/Fig8ab_shrink.svg".format())
    plt.show()
    df_FS_FP_IS_IP = pd.DataFrame(FS_FP_IS_IP)
    df_FS_FP_IS_IP.to_csv("./figures/RAL_plots/WilcoxonSignedRank/FS_FP_IS_IP.csv")

def plot_theta_error_merged_bestLoss8_onesidedtest(version, model_name_list, str_list):
    plt.figure(figsize=(88.9 / 25.4*2, 88.9 / 25.4 / 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    FS_FP_IS_IP = np.empty((16, 16), dtype='<U10')
    for number, model_type in enumerate(["forward", "inverse"]):
        unit = "degrees" if model_type == "forward" else "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        old_new_loss = []
        rmse_array_list = []  # data for paired t-test
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                model_list = [model_type + str + train_data for str in str_list]  #
                for test_data in ["sinus", "stair1_10s"]:
                    if train_data=="Sinus12" and test_data=="stair1_10s":
                        continue
                    if train_data=="Stair5s" and test_data=="sinus":
                        continue
                    for model_name in model_list:
                        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name, test_data))
                        rmse_me_StdL = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}StdL/{}_error.npy".format(model_name, test_data))
                        if np.mean(rmse_me[:, 0]) <= np.mean(rmse_me_StdL[:, 0]):
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                            me_dfs.append(np.max(rmse_me[:, 1]))
                            old_new_loss.append(0)
                            rmse_array_list.append(rmse_me[:, 0])
                        else:
                            rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me_StdL[:, 0]))
                            me_dfs.append(np.max(rmse_me_StdL[:, 1]))
                            old_new_loss.append(1)
                            rmse_array_list.append(rmse_me_StdL[:, 0])

        print(rmse_mean_dfs)
        print(me_dfs)
        tick_list = [i + 1 for i in range(len(model_name_list)*2)]
        plt.subplot(1, 2, number+1)
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        symbol_list = ["FNN", "BK", "PI", "LSTM", "GRU", "FNN-HIB", "HB1", "HB2"]
        symbol_list2 = ["FNN", "Backlash", "PI", "LSTM", "GRU", "FNN-HIB", "HB1: Serial LSTM-Backlash", "HB2: Parallel LSTM-Backlash"]

        # do the paired t-test and show the result in a plot
        print(model_type)

        m = []
        for i in range(8):
            n = []
            rmse_array1 = rmse_array_list[i]
            for j in range(8):
                if i == j:
                    n.append(np.nan)
                    if model_type == "forward":
                        FS_FP_IS_IP[i * 2, j * 2] = "_"
                    else:
                        FS_FP_IS_IP[i * 2, j * 2+1] = "_"
                    continue
                rmse_array2 = rmse_array_list[j]
                # t_stat, p_value = stats.ttest_rel(rmse_array1, rmse_array2)
                statistic, p_value_t = stats.wilcoxon(rmse_array1, rmse_array2)
                statistic, p_value_g = stats.wilcoxon(rmse_array1, rmse_array2, alternative="greater")
                statistic, p_value_l = stats.wilcoxon(rmse_array1, rmse_array2, alternative="less")
                # n.append(round(p_value, 4))
                # n.append(p_value)
                n.append(f'{p_value_g:.{3-1}e}')
                if model_type == "forward":
                    if p_value_g < 0.05:
                        FS_FP_IS_IP[i*2, j*2] = ">"
                    elif p_value_l < 0.05:
                        FS_FP_IS_IP[i * 2, j * 2] = "<"
                    else:
                        if p_value_t < 0.05:
                            FS_FP_IS_IP[i*2, j*2] = "n"
                        else:
                            FS_FP_IS_IP[i * 2, j * 2] = "~"
                else:
                    if p_value_g < 0.05:
                        FS_FP_IS_IP[i*2, j*2+1] = ">"
                    elif p_value_l < 0.05:
                        FS_FP_IS_IP[i*2, j*2+1] = "<"
                    else:
                        if p_value_t < 0.05:
                            FS_FP_IS_IP[i * 2, j * 2 + 1] = "n"
                        else:
                            FS_FP_IS_IP[i * 2, j * 2 + 1] = "~"
                # print(f"Paired T-Test: T-statistic: {t_stat}, P-value: {p_value}")
            m.append(n)

        df = pd.DataFrame(m, columns=symbol_list, index=symbol_list)
        df.to_csv("./figures/RAL_plots/WilcoxonSignedRank_onesided/{}_sinusoidal.csv".format(model_type))

        m = []
        for i in range(8, 16):
            n = []
            rmse_array1 = rmse_array_list[i]
            for j in range(8, 16):
                if i == j:
                    n.append(np.nan)
                    if model_type == "forward":
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = "_"
                    else:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "_"
                    continue
                rmse_array2 = rmse_array_list[j]
                # t_stat, p_value = stats.ttest_rel(rmse_array1, rmse_array2)
                statistic, p_value_t = stats.wilcoxon(rmse_array1, rmse_array2)
                statistic, p_value_g = stats.wilcoxon(rmse_array1, rmse_array2, alternative="greater")
                statistic, p_value_l = stats.wilcoxon(rmse_array1, rmse_array2, alternative="less")
                # print(f"Paired T-Test: T-statistic: {t_stat}, P-value: {p_value}")
                # n.append(round(p_value, 4))
                # n.append(p_value)
                n.append(f'{p_value_g:.{3-1}e}')
                if model_type == "forward":
                    if p_value_g < 0.05:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = ">"
                    elif p_value_l < 0.05:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = "<"
                    else:
                        if p_value_t < 0.05:
                            FS_FP_IS_IP[(i - 8) * 2 + 1, (j - 8) * 2] = "n"
                        else:
                            FS_FP_IS_IP[(i-8)*2+1, (j-8)*2] = "~"
                else:
                    if p_value_g < 0.05:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = ">"
                    elif p_value_l < 0.05:
                        FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "<"
                    else:
                        if p_value_t < 0.05:
                            FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "n"
                        else:
                            FS_FP_IS_IP[(i-8)*2+1, (j-8)*2+1] = "~"

            m.append(n)
        df = pd.DataFrame(m, columns=symbol_list, index=symbol_list)
        df.to_csv("./figures/RAL_plots/WilcoxonSignedRank_onesided/{}_point_to_point.csv".format(model_type))

        # show the maximum value and the model name
        if model_type == "forward":
            offset = 0.12
        else:
            offset = 0.012
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            if old_new_loss[i] == 1:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i]-offset, symbol_list[i%8], ha='center', va='bottom', fontsize=6) #, fontweight='bold')
            else:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 8], ha='center', va='bottom', fontsize=6)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        # plt the error bar
        if version == "v1":
            # fmt_list = ['b'+f for f in ['o', '^', 'p', 'h', '*', '+', 'x']]
            color_list = ['brown', 'red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 8):
                for j in range(0, 2):
                    if j == 0:
                        plt.errorbar(tick_list[i+8*j], rmse_mean_dfs[i+8*j], yerr=rmse_std_dfs[i+8*j], fmt="o", color=color_list[i], label=symbol_list2[i], capsize=5) #color=color_list[i],
                    else:
                        plt.errorbar(tick_list[i+8*j], rmse_mean_dfs[i+8*j], yerr=rmse_std_dfs[i+8*j], fmt="o", color=color_list[i], capsize=5) #color=color_list[i],
            for x in np.arange(1, 2)*8+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 2) * 8 + 4, ["Sinusoids", "Point-to-point"])

        # if model_type == "forward":
        #     plt.ylim([0, 3.2])
        # else:
        #     plt.ylim([0, 0.37])
        # plt.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.ylabel("RMS Error / Maximum Error ({})".format(unit))
    plt.tight_layout()
    # plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/BestLoss_2groups_{}_{}.png".format(model_type, version), dpi=600)
    # plt.savefig("./figures/RAL_plots/Fig8ab.png".format(), dpi=600)
    # plt.savefig("./figures/RAL_plots/Fig8ab.svg".format())
    plt.show()
    df_FS_FP_IS_IP = pd.DataFrame(FS_FP_IS_IP)
    df_FS_FP_IS_IP.to_csv("./figures/RAL_plots/WilcoxonSignedRank_onesided/FS_FP_IS_IP.csv")

def plot_Y_error_merged_BestLoss8(version, model_name_list, str_list):
    plt.figure(figsize=(88.9 / 25.4 * 2, 88.9 / 25.4 / 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'

    for number, model_type in enumerate(["forward", "inverse"]):
        unit = "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        old_new_loss = []
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                model_list = [model_type + str + train_data for str in ["_FEED_"]]
                model_list = model_list + [model_type + str + train_data + "_map" for str in ["_bkl_", "_PI_"]]
                model_list = model_list + [model_type + str + train_data for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
                model_StdL_list = [model_type + str + train_data + "StdL" for str in ["_FEED_"]]
                model_StdL_list = model_StdL_list + [model_type + str + train_data + "StdL" + "_map" for str in ["_bkl_", "_PI_"]]
                model_StdL_list = model_StdL_list + [model_type + str + train_data + "StdL" for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
                for test_data in ["sinus", "stair1_10s"]:
                    if train_data=="Sinus12" and test_data=="stair1_10s":
                        continue
                    if train_data=="Stair5s" and test_data=="sinus":
                        continue
                    for model_name, model_name_StdL in zip(model_list, model_StdL_list):
                        rmse_me = np.load(
                            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name,
                                                                                                       test_data))
                        rmse_me_StdL = np.load(
                            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name_StdL,
                                                                                                           test_data))

                        if rmse_me.shape[1] == 4:
                            if np.mean(rmse_me[:, 1]) <= np.mean(rmse_me_StdL[:, 1]):
                                rmse_mean_dfs.append(np.mean(rmse_me[:, 1]))
                                rmse_std_dfs.append(np.std(rmse_me[:, 1]))
                                me_dfs.append(np.max(rmse_me[:, 3]))
                                old_new_loss.append(0)
                            else:
                                rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 1]))
                                rmse_std_dfs.append(np.std(rmse_me_StdL[:, 1]))
                                me_dfs.append(np.max(rmse_me_StdL[:, 3]))
                                old_new_loss.append(1)
                        else:
                            if np.mean(rmse_me[:, 0]) <= np.mean(rmse_me_StdL[:, 0]):
                                rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                                rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                                me_dfs.append(np.max(rmse_me[:, 1]))
                                old_new_loss.append(0)
                            else:
                                rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 0]))
                                rmse_std_dfs.append(np.std(rmse_me_StdL[:, 0]))
                                me_dfs.append(np.max(rmse_me_StdL[:, 1]))
                                old_new_loss.append(1)


        tick_list = [i + 1 for i in range(len(model_name_list)*2)]
        plt.subplot(1, 2, number+1)
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        # plt.plot(tick_list, me_dfs, 'r*')
        symbol_list = ["FNN", "BK", "PI", "LSTM", "GRU", "FNN-HIB", "HB1", "HB2"]
        if model_type == "forward":
            offset = 0.051
        else:
            offset = 0.012
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            if old_new_loss[i] == 1:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 8], ha='center', va='bottom', fontsize=6) #, fontweight='bold')
            else:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 8], ha='center', va='bottom', fontsize=6)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        if version == "v1":
            color_list = ['brown', 'red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 8):
                for j in range(0, 2):
                    plt.errorbar(tick_list[i + 8 * j], rmse_mean_dfs[i + 8 * j], yerr=rmse_std_dfs[i + 8 * j], fmt="o",
                                 color=color_list[i], label='Mean ± std', capsize=5)
            for x in np.arange(1, 2)*8+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 2) * 8 + 4, ["Sinusoids", "Point-to-point"])
        # if model_type == "forward":
        #     plt.ylim([0, 2.5])
        # else:
        #     plt.ylim([0, 0.37])
        plt.ylabel("RMS Error / Maximum Error ({})".format(unit))
        plt.tight_layout()
        # plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/tipD/BestLoss_2groups_Y_{}_{}.png".format(model_type, version), dpi=600)
    plt.savefig("./figures/RAL_plots/Fig8cd.png".format(), dpi=600)
    plt.savefig("./figures/RAL_plots/Fig8cd.svg".format())
    plt.show()

def plot_figure8():
    version = "v1"
    model_name_list = ["FNN", "Backlash", "RDPI", "LSTM", "GRU", "FNN-HIB", "LSTM-Backlash",
                       "LSTM-Backlash-sum"]  # "LSTM-Backlash-sum"
    str_list = ["_FEED_", "_bkl_", "_PI_", "_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]
    plot_theta_error_merged_bestLoss8(version, model_name_list, str_list)
    # plot_theta_error_merged_bestLoss8_onesidedtest(version, model_name_list, str_list)
    # plot_Y_error_merged_BestLoss8(version, model_name_list, str_list)

def plot_figure9():
    return 0

if __name__ == '__main__':
    # Fig 3
    # plot_figure3()


    #  Fig. 5: (a) Bending angle vs tendon displacement (5 frequencies),
    #  (b) Bending angle vs tendon force (5 frequencies),
    #  (c) Bending angle vs tendon displacement (5 trials at same frequency).
    #  Parts (a) and (c) should use same vertical scale.
    plot_figure5()

    # Fig. 6: (a) Alternative mappings in block diagram form, (b) Backlash model map to X – overplot of direct and two-part maps.
    #
    # plot_figure6()

    # •	Fig. 7: Effect of training motion on model performance.
    # (a) Sinusoidal tip angle vs time – overplot models (one type of model) trained on sinusoidal vs stop and go motions.
    # (b) Stop-and-go tip angle vs time – overplot models (one type of model) trained on sinusoidal vs stop and go motions.
    # plot_figure7()

    # Fig.8
    # plot_figure8()

    # Fig.9
    # plot_figure9()
