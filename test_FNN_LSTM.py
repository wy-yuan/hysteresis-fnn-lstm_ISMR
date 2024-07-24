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
# from plot_compare_models import test_LSTM
from plot_compare_models import test_FNN
from models.hybrid_model import LSTM_backlash_Net
from models.hybrid_model import BacklashNet
from models.hybrid_model import LSTM_backlash_sum3_Net
from models.hybrid_model import fixedBacklash
from models.hybrid_model import backlash_LSTM_Net
from models.hybrid_model import BacklashInverseNet
from models.hybrid_model import LSTMBacklashInverseNet
from models.hybrid_model import LSTMBacklashSumInvNet
from models.hybrid_model import fixedBacklashInverseNet
from train_TC import GRUNet
from models.BoucWen_model import BoucWenNet
from models.GRDPI_model import GRDPINet
from models.GRDPI_model import GRDPI_inv_Net


def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

def load_data(data_path, seg, sample_rate=100):
    data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    time = data[:, 0]
    tendon_disp = data[:, 1]
    tip_A = data[:, 8]
    em_pos = data[:, 5:8]
    # tip_disp = np.linalg.norm(em_pos, axis=1)
    tip_disp = data[:, 5]
    # resample data using fixed sampling rate
    # sample_rate = 100  # Hz
    frequency = round(1 / (time[-1] / 7), 2)
    interp_time = np.arange(10, time[-1], 1 / sample_rate)
    tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
    tip_A_resample = np.interp(interp_time, time, tip_A)
    tip_disp_resample = np.interp(interp_time, time, tip_disp)

    # normalization to [0, 1] and pad -1
    tendon_disp = np.hstack([np.ones(seg), tendon_disp_resample / 6]) #6
    tip_A = np.hstack([np.ones(seg), tip_A_resample / 90]) #90
    tip_D = tip_disp_resample / 60
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'tip_D': tip_D, 'freq': freq}

def test_PI(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    if forward:
        model = GRDPINet()
    else:
        model = GRDPI_inv_Net()
    sr = sr
    seg = 0
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()
    tip_name = 'tip_A'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name] = data_filter(data[tip_name])

    joints = data['tendon_disp'][:, np.newaxis, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, np.newaxis, np.newaxis].astype("float32")
    tips = data[tip_name][:, np.newaxis, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis, np.newaxis].astype("float32")
    scale = 1
    if forward:
        joints_ = torch.tensor(joints).to(device)*scale
    else:
        joints_ = torch.tensor(tips).to(device)
        # joints_ = (torch.tensor(tips).to(device)-0.33)*1.5

    out = []
    h_init = joints_[0:1, :, :].expand(1, 1, 4)
    # h_init = torch.ones(1, 1, 4).to(device)
    print(h_init)
    h_ = h_init
    for i in range(data['tendon_disp'].shape[0]):
        input_ = joints_[i:i + 1, 0:input_dim]
        if i == 0:
            input_pre = joints_[i:i + 1, 0:input_dim]
        else:
            input_pre = joints_[i-1:i, 0:input_dim]
        output, weights, h_ = model(input_, input_pre, h_)
        print("weights--------------", weights)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)
    if forward:
        out = out/scale
        input = data['tendon_disp'][seg:]*6
        gt = data[tip_name][seg:]*90
        output = out[seg:, 0]*90
    else:
        input = data[tip_name][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def test_PI_test(pt_path, forward=True):
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_rmbNorm/TC_GRDPI_bs16_epoch124_best0.0010092476692634745.pt"
    inverse_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_gamma_InitRandn/TC_GRDPI_bs16_epoch6.pt"
    device = "cuda"
    model = GRDPINet()
    model.load_state_dict(torch.load(forward_PI_path, map_location=device))
    model.cuda()
    model.eval()

    A = 0.5
    f = 0.1
    x = np.linspace(0, 20, 500)
    tau = (-f / 6) * np.log(2 / 14)
    u = A * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2)) + A
    u_ = u[:, np.newaxis, np.newaxis]

    u_ = torch.tensor(u_).to(device)
    out = []
    h_init = u_[0:1, :, :].expand(1, 1, 4)
    h_ = h_init
    for i in range(u_.shape[0]):
        input_ = u_[i:i + 1, 0:input_dim]
        if i == 0:
            input_pre = u_[i:i + 1, 0:input_dim]
        else:
            input_pre = u_[i - 1:i, 0:input_dim]
        output, weights, h_ = model(input_, input_pre, h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)


    model = GRDPI_inv_Net()
    model.load_state_dict(torch.load(inverse_PI_path, map_location=device))
    model.cuda()
    model.eval()
    inv_out = []
    out_ = out[:, np.newaxis]
    print(out_.shape)
    out_ = torch.tensor(out_).to(device)
    h_init = out_[0:1, :, :].expand(1, 1, 4)
    h_ = h_init
    for i in range(out_.shape[0]):
        input_ = out_[i:i + 1, 0:input_dim]
        if i == 0:
            input_pre = out_[i:i + 1, 0:input_dim]
        else:
            input_pre = out_[i - 1:i, 0:input_dim]
        output, weights, h_ = model(input_, input_pre, h_)
        pre_pos = output.detach().cpu().numpy()[0]
        inv_out.append(pre_pos[0])
    inv_out = np.array(inv_out)

    plt.subplot(1, 2, 1)
    plt.plot(u, out)
    plt.subplot(1, 2, 1)
    plt.plot(out, inv_out)
    plt.subplot(1, 2, 2)
    plt.plot(u, inv_out)
    plt.plot(u, u, 'r--')
    plt.show()


def test_BoucWen(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    model = BoucWenNet()
    sr = sr
    seg = 1
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()
    tip_name = 'tip_A'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name] = data_filter(data[tip_name])

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, np.newaxis].astype("float32")
    tips = data[tip_name][:, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    if forward:
        joints_ = joints
    else:
        joints_ = tips

    out = []
    h_init = torch.zeros(1, 1, 1).to(device)
    h_ = h_init
    for i in range(data['tendon_disp'].shape[0]):
        input_ = joints_[i:i + 1, 0:input_dim]
        if i == 0:
            input_pre = joints_[i:i + 1, 0:input_dim]
        else:
            input_pre = joints_[i-1:i, 0:input_dim]
        output, weights, h_ = model(torch.tensor([input_]).to(device), torch.tensor([input_pre]).to(device), h_)
        print("weights--------------", weights)
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
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def test_LSTM(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    model = LSTMNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()
    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name] = data_filter(data[tip_name])

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
        gt = data[tip_name][seg:]*60
        output = out[seg:, 0]*60
    else:
        input = data[tip_name][seg:] * 60
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def test_GRU(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    model = GRUNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    data = load_data(data_path, seg, sample_rate=sr)
    data['tip_A'] = data_filter(data["tip_A"])

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
    if fq:
        freq = data['freq'][:, np.newaxis].astype("float32")
        joints = np.concatenate([joints, freq], axis=1)

    out = []
    hidden = torch.zeros(model.num_layers, 1, model.hidden_dim).to(device)
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
        gt = data['tip_A'][seg:]*90
        output = out[seg:, 0]*90
    else:
        input = data['tip_A'][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def test_LSTMbacklash(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    if forward:
        model = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    else:
        model = LSTMBacklashInverseNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = 0
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name] = data_filter(data[tip_name])

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, np.newaxis].astype("float32")
    tips = data[tip_name][:, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    out = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        tip = tips[i:i + 1, 0:input_dim]
        # pre_pos_ = pre_pos[i-1:i, 0:input_dim]
        if forward:
            output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = model(
                torch.tensor([input_]).to(device), torch.tensor([pre_pos]).to(device), h_)
            # print(i, dynamic_weights, condition_lo, condition_up, condition_mid)
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        else:
            output, h_, dynamic_weights = model(torch.tensor([tip]).to(device), torch.tensor([pre_pos]).to(device),
                                                torch.tensor([pre_td]).to(device), h_)  # x_curr, _x_prev, u_pre
            pre_td = output.detach().cpu().numpy()[0]
            out.append(pre_td[0])
            pre_pos = tip
    out = np.array(out)
    if forward:
        input = data['tendon_disp'][seg:] * 6
        gt = data[tip_name][seg:] * 60
        output = out[seg:, 0] * 60
    else:
        input = data[tip_name][seg:] * 60
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def Forward_LSTMbacklash(pt_path, data_td, data_tip, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    model = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl)

    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    joints = data_td[:, np.newaxis].astype("float32")/6
    pre_pos = data_tip[0:1, np.newaxis].astype("float32")/90

    out = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(0, data_td.shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = model(torch.tensor([input_]).to(device), torch.tensor([pre_pos]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)
    gt = data_tip
    output = out[:, 0] * 90

    res_rmse = rmse_norm(output, gt)
    res_nrmse = res_rmse / (max(gt) - min(gt))
    return output, res_rmse, res_nrmse


import copy
def data_filter(data, datatype="angle"):
    if datatype == "angle":
        # 小范围的上升或下降，就把这个值拉到和前一个值相同
        filtered = copy.deepcopy(data)
        for i in range(1, data.shape[0]-1):
            v = abs(data[i]-filtered[i-1])
            if v < 0.0003:  # v should be larger than 0.0001,
                filtered[i] = filtered[i-1]
    elif datatype == "disp":
        filtered1 = copy.deepcopy(data)
        n = 1
        for i in range(0, data.shape[0]):
            filtered1[i] = np.mean(data[max(i-n, 0):i+n+1])
        filtered = copy.deepcopy(filtered1)
        for i in range(1, data.shape[0] - 1):
            v = abs(data[i] - filtered[i - 1])
            if v < 0.0015:  # v should be larger than 0.0001,
                filtered[i] = filtered[i - 1]
        for j in range(0):
            filtered2 = copy.deepcopy(filtered)
            n = 1
            for i in range(0, data.shape[0]):
                arr = filtered[max(i - n, 0):i + n + 1]
                if len(arr)<2*n+1:
                    filtered2[i] = np.mean(arr)
                else:
                    filtered2[i] = np.sum(arr*np.array([1,2,1])/4)
            # for i in range(1, data.shape[0] - 1):
            #     v = abs(data[i] - filtered[i - 1])
            #     if v < 0.001:  # v should be larger than 0.0001,
            #         filtered[i] = filtered[i - 1]
    # plt.plot(data)
    plt.plot(filtered*60)
    plt.show()
    return filtered

def test_backlash(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, rm_init=0):
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

    data = load_data(data_path, seg, sample_rate=sr)
    data['tip_A'] = data_filter(data["tip_A"])

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data['tip_A'][0:1, np.newaxis].astype("float32")
    tips = data['tip_A'][:, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    out = []
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        tip = tips[i:i + 1, 0:input_dim]
        if forward:
            output, dynamic_weights, condition_lo, condition_up, condition_mid = model(torch.tensor([joint]).to(device),
                                                                                       torch.tensor([pre_pos]).to(
                                                                                           device))
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        else:
            output, dynamic_weights = model(torch.tensor([tip]).to(device), torch.tensor([pre_pos]).to(device),
                                            torch.tensor([pre_td]).to(device))  # x_curr, _x_prev, u_pre
            pre_td = output.detach().cpu().numpy()[0]
            out.append(pre_td[0])
            pre_pos = tip
    print(dynamic_weights)
    out = np.array(out)
    if forward:
        input = data['tendon_disp'][seg:] * 6
        gt = data['tip_A'][seg:] * 90
        output = out[seg:, 0] * 90
    else:
        input = data['tip_A'][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse


def test_LSTM_backlash_sum3(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    if forward:
        model = LSTM_backlash_sum3_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    else:
        model = LSTMBacklashSumInvNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = 0
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    fixed_backlash = fixedBacklash()
    fixed_backlash.cuda()
    fixed_backlash.eval()

    fixed_backlashInv = fixedBacklashInverseNet()
    fixed_backlashInv.cuda()
    fixed_backlashInv.eval()

    data = load_data(data_path, seg, sample_rate=sr)
    data['tip_A'] = data_filter(data["tip_A"])
    if forward:
        out_bl = torch.tensor([data['tip_A'][0]]).to(device)
    else:
        out_bl = torch.tensor([data['tendon_disp'][0]]).to(device)

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data['tip_A'][0:1, np.newaxis].astype("float32")
    tips = data['tip_A'][:, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    out = []
    out1 = []
    out2 = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        tip = tips[i:i + 1, 0:input_dim]
        if forward:
            out_bl = fixed_backlash(torch.tensor([input_]).to(device), out_bl)
            output, h_, out_l = model(torch.tensor([input_]).to(device), out_bl, h_)
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        else:
            out_bl = fixed_backlashInv(torch.tensor([tip]).to(device), torch.tensor([pre_pos]).to(device), out_bl)
            output, h_, out_l = model(torch.tensor([tip]).to(device), out_bl, h_)
            pre_td = output.detach().cpu().numpy()[0]
            out.append(pre_td[0])
            pre_pos = tip

        out1.append(out_bl.detach().cpu().numpy()[0, 0])
        out2.append(out_l.detach().cpu().numpy()[0, 0])
    out = np.array(out)
    out1 = np.array(out1)
    out2 = np.array(out2)
    if forward:
        input = data['tendon_disp'][seg:] * 6
        gt = data['tip_A'][seg:] * 90
        output = out[seg:, 0] * 90
    else:
        input = data['tip_A'][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    print(output.shape, out1.shape)
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    plt.plot(out1[seg:, 0])
    plt.plot(out2[seg:, 0])
    plt.show()
    # plt.plot(input, out1[seg:, 0]*90)
    # plt.plot(input, output)
    # plt.show()
    return input, gt, output, out1[seg:, 0] * 90, data['time'], res_rmse, res_nrmse


def test_backlash_LSTM(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    model = backlash_LSTM_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = 0
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    fixed_backlash = fixedBacklash()
    fixed_backlash.cuda()
    fixed_backlash.eval()

    data = load_data(data_path, seg, sample_rate=sr)
    out_bl = torch.tensor([data['tip_A'][0].astype("float32")]).to(device)
    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
        pre_pos = 0
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
        pre_pos = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    out = []
    out1 = []
    out2 = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        out_bl = fixed_backlash(torch.tensor([input_]).to(device), out_bl)
        output, h_, out_l = model(torch.tensor([input_]).to(device), out_bl, h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
        out1.append(out_bl.detach().cpu().numpy()[0, 0])
        out2.append(out_l.detach().cpu().numpy()[0, 0])
    out = np.array(out)
    out1 = np.array(out1)
    out2 = np.array(out2)
    if forward:
        input = data['tendon_disp'][seg:] * 6
        gt = data['tip_A'][seg:] * 90
        output = out[seg:, 0] * 90
    else:
        input = data['tip_A'][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    print(output.shape, out1.shape)
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    # plt.plot(out1[seg:, 0])
    # plt.plot(out2[seg:, 0])
    # plt.show()
    # plt.plot(input, out1[seg:, 0]*90)
    # plt.plot(input, output)
    # plt.show()
    return input, gt, output, out1[seg:, 0] * 90, data['time'], res_rmse, res_nrmse


def save_backlash(data_path, out_path, data_time, predicted_tip):
    ori_data = np.loadtxt(data_path, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    ori_data[:, 8] = data_filter(ori_data[:, 8])
    predicted_tip_rs = np.interp(ori_data[:, 0], data_time, predicted_tip)
    label = ["Time", "Tendon1 Disp.", "Tendon2 Disp.", "Motor1 Curr.", "Motor2 Curr.", "Pos x", "Pos y", "Pos z",
             "Angle x", "Angle y", "Angle z", "backlash_out"]
    data_to_save = np.hstack([ori_data, predicted_tip_rs[:, np.newaxis]])
    data_to_save = np.row_stack((label, data_to_save))
    np.savetxt(out_path, data_to_save, delimiter=',', fmt='%s')
    print(data_to_save.shape)


if __name__ == '__main__':
    forward_LSTM_path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch488_best0.00010981084632440833.pt"
    inverse_LSTM_path = "./checkpoints/Train12345Hz1rep_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch459_best0.00019231367984396563.pt"
    inverse_FNN_path = "./checkpoints/Train12345Hz1repFakePause_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch466_best0.00019089564545550264.pt"
    inverse_FNN_path = "./checkpoints/Train12345Hz1repFakePause_INV_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch456_best0.00021606441196008862.pt"

    # forward_LSTM_path = "./checkpoints/longPause_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch397_best0.0002755442473537218.pt"
    # forward_FNN_path = "./checkpoints/longPause_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch473_best0.00019102727674180642.pt"
    # inverse_LSTM_path = "./checkpoints/longPause_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch417_best0.00024064954666724682.pt"
    # inverse_FNN_path = "./checkpoints/longPause_INV_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch453_best0.000295662209791155.pt"
    # forward_LSTM_path = "./checkpoints/longPause_12345Hz_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch465_best0.0002790999646156521.pt"
    # inverse_LSTM_path = "./checkpoints/longPause_12345Hz_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch475_best0.0003266395310751928.pt"
    # inverse_LSTM_path = "./checkpoints/longPauseOnly_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch449_best0.00023493068406192617.pt"
    forward_LSTM_path = "./checkpoints/innerTSlackSinus_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch496_best0.00015579808262243335.pt"
    inverse_LSTM_path = "./checkpoints/innerTSlackSinus_INV_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch477_best0.00022321039455525194.pt"
    forward_FNN_path = "./checkpoints/innerTSlackSinus_FNN_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch358_best7.616092859409816e-05.pt"
    inverse_FNN_path = "./checkpoints/innerTSlackSinus_INV_FNN_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch328_best0.00019501733544212397.pt"

    # forward_LSTM_path = "./checkpoints/innerTSlack20_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch424_best0.00016414528836899007.pt"
    # forward_FNN_path = "./checkpoints/innerTSlack20_FNN_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch429_best0.00010219023637806199.pt"

    forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch489_best0.00018550869560660056.pt"
    # inverse_LSTM_path = "./checkpoints/innerTSlack40_INV_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch370_best0.0002499642580496295.pt"
    # forward_FNN_path = "./checkpoints/innerTSlack40_FNN_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch269_best8.459922127445997e-05.pt"
    # inverse_FNN_path = "./checkpoints/innerTSlack40_INV_FNN_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch490_best0.0002087136608246337.pt"

    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_seg50_sr25_NOrsNOfq_lr1e-4/TC_LSTM_L2_bs16_epoch407_best0.0002356159649992598.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_seg50_sr25_NOrsNOfq_lr1e-5/TC_LSTM_L2_bs16_epoch478_best0.00038841567676706565.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_seg50_sr25_NOrsNOfq_wde-5/TC_LSTM_L2_bs16_epoch407_best0.0003942580877928921.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_seg50_sr25_NOrsNOfq_wde-6/TC_LSTM_L2_bs16_epoch478_best0.00027909159470009546.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq/TC_LSTM_L1_bs16_epoch184_best0.00021742242097388953.pt"  # 139
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_h16_seg50_sr25_NOrsNOfq/TC_LSTM_L1_bs16_epoch46_best0.0003348217994373824.pt"
    forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_std0.01/TC_LSTM_L1_bs16_epoch500.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_std0.02/TC_LSTM_L1_bs16_epoch221_best-0.0004814212235422539.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_std0.015/TC_LSTM_L1_bs16_epoch221_best-0.00028297034880360737.pt"
    forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_stdRatio0.0001/TC_LSTM_L1_bs16_epoch350.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_stdRatio0.001/TC_LSTM_L1_bs16_epoch412_best0.0007990132176524235.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_stdRatio0.0005/TC_LSTM_L1_bs16_epoch200.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_stdRatio0.0002/TC_LSTM_L1_bs16_epoch470.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_stdRatio0.00005/TC_LSTM_L1_bs16_epoch500.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_stdRatio1e-4_delta1e-5/TC_LSTM_L1_bs16_epoch216_best0.0005631793709841394.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_stdRatio1e-5_delta1e-5/TC_LSTM_L1_bs16_epoch330.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_stdRatio1e-3_delta1e-3/TC_LSTM_L1_bs16_epoch100.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_stdRatio1e-4_delta1e-3/TC_LSTM_L1_bs16_epoch500.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_stdRatio0.00015_delta1e-4/TC_LSTM_L1_bs16_epoch450.pt"
    # forward_LSTM_path = "./checkpoints/innerTSlack40_LSTM_layer2_seg50_sr25_stdRatio1e-4_delta1e-4/TC_LSTM_L2_bs16_epoch229_best0.0003166038428418457.pt"

    forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_BacklashNet_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch30.pt"
    forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMbacklash_layer2_seg100_sr25/TC_LSTMbacklash_L2_bs16_epoch97_best6.848830003450116e-05.pt"
    forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMbacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch250.pt"
    # forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMbacklash_layer2_seg50_sr25_detach/TC_LSTMbacklash_L2_bs16_epoch250.pt"
    # forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMbacklash_layer2_seg20_sr25/TC_LSTMbacklash_L2_bs16_epoch300.pt"
    # forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMBLsum_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch248_best1.98890256257782e-05.pt"
    # forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMBLsum2_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch300.pt"
    # forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_LSTMBLsum3_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch300.pt"
    # forward_LSTMbacklash_path = "./checkpoints/innerTSlackSinus_BL-LSTM_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch288_best9.863676148763096e-05.pt"

    forward_backlash_path = "./checkpoints/Sinus_BacklashNet_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch73_best8.678210576625256e-05.pt"
    forward_LSTM_path = "./checkpoints/Sinus_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch248_best6.121114939372683e-05.pt"
    forward_LSTM_path = "./checkpoints/SinusRandom20_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch54_best0.00017889751063194126.pt"
    # forward_LSTM_path = "./checkpoints/Random20_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch200.pt"
    forward_LSTM_path = "./checkpoints/TipDStair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch56_best0.00024147247095243073.pt"

    forward_LSTMbacklash_path = "./checkpoints/Sinus_LSTMbacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch135_best2.0372903005438044e-05.pt"
    # forward_LSTMbacklash_path = "./checkpoints/SinusRandom_LSTMbacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch121_best2.5740021700339785e-05.pt"
    # forward_LSTMbacklash_path = "./checkpoints/SinusRandom_LSTMbacklash_layer2_seg100_sr25/TC_LSTMbacklash_L2_bs16_epoch120_best2.249701772788244e-05.pt"
    # forward_LSTMbacklash_path = "./checkpoints/Random20_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch200.pt"
    forward_LSTMbacklash_path = "./checkpoints/SinusRandom20_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch14_best4.934203727316344e-05.pt"
    forward_LSTMbacklash_path = "./checkpoints/Stair_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch80_best2.0286842234496316e-05.pt"
    forward_LSTMbacklash_path = "./checkpoints/SinusStair_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch86_best2.800920207475591e-05.pt"
    forward_LSTMbacklash_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch57_best3.1443552133764724e-05.pt"
    forward_LSTMbacklash_path = "./checkpoints/TipDStair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch75_best2.5949533892344334e-05.pt"
    forward_LSTMbacklash_path = "./checkpoints/TipDXStair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch32_best1.4133408357489923e-05.pt"

    forward_LSTMBacklashSum_path = "./checkpoints/Sinus_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch280_best2.5839712842583648e-05.pt"
    forward_LSTMBacklashSum_path = "./checkpoints/SinusRandom_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch75_best3.9291964855954695e-05.pt"
    forward_backlashLSTM_path = "./checkpoints/Sinus_BacklashLSTM_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch245_best3.0863524844497045e-05.pt"
    forward_backlashLSTM_path = "./checkpoints/SinusRandom_BacklashLSTM_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch241_best4.035687999345039e-05.pt"

    inverse_backlashInv_path = "./checkpoints/Sinus_BacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch236_best0.00036487386042007513.pt"
    inverse_LSTMBacklashInv_path = "./checkpoints/Sinus_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch283_best7.757545584246469e-05.pt"
    inverse_LSTMBacklashInv_path = "./checkpoints/SinusRandom_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch154_best0.000103270593202886.pt"
    inverse_LSTMBacklashSumInv_path = "./checkpoints/Sinus_LSTMBacklashSumInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch55_best0.0001412896214741661.pt"
    # inverse_LSTMBacklashSumInv_path = "./checkpoints/SinusRandom_LSTMBacklashSumInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch278_best0.00012105015072955797.pt"
    inverse_LSTM_path = "./checkpoints/Sinus_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch112_best0.00017308813514349728.pt"
    inverse_LSTM_path = "./checkpoints/SinusRandom_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch122_best0.0002118320615224193.pt"

    forward_GRU_path = "./checkpoints/Sinus_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch105_best6.201170519057472e-05.pt"
    # forward_GRU_path = "./checkpoints/SinusRandom_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch101_best0.00010656737430184937.pt"

    forward_BoucWen_path = "./checkpoints/Sinus_BoucWen_seg50_sr25_lre-1_7w/TC_BoucWen_bs16_epoch233_best0.00027222971180828796.pt"
    forward_BoucWen_path = "./checkpoints/Sinus_BoucWen_seg50_sr25_7w/TC_BoucWen_bs16_epoch139_best0.00025414203523492474.pt"
    forward_BoucWen_path = "./checkpoints/Sinus_BoucWen_seg50_sr25_6w/TC_BoucWen_bs16_epoch222_best0.0002502208029046276.pt"
    # forward_BoucWen_path = "./checkpoints/Sinus_BoucWen_seg50_sr25_6w_lre-2/TC_BoucWen_bs16_epoch109_best0.0002622656770577354.pt"
    inverse_BoucWen_path = "./checkpoints/Sinus_inverse_BoucWen_seg50_sr25_6w/TC_BoucWen_bs16_epoch269_best0.0003674909042320767.pt"

    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_losslast/TC_GRDPI_bs16_epoch7_best0.0005771191233287612.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_losslast_relu/TC_GRDPI_bs16_epoch14_best0.00010664762743606659.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu/TC_GRDPI_bs16_epoch111_best0.00039858848346847154.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_gamma/TC_GRDPI_bs16_epoch3_best0.0008598278460124413.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_gamma_rep2/TC_GRDPI_bs16_epoch15_best0.0004877381157958722.pt"
    # forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_gamma_InitRandn/TC_GRDPI_bs16_epoch107_best0.00039200111681011394.pt"
    # forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_rmb/TC_GRDPI_bs16_epoch49_best0.0004974768484224391.pt"
    # forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_rmbNorm/TC_GRDPI_bs16_epoch124_best0.0010092476692634745.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_SmallBetaMax50/TC_GRDPI_bs16_epoch59_best1.318392085827003.pt"
    # forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_SmallBetaMax/TC_GRDPI_bs16_epoch49_best0.0004439635707412736.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_SmallBetaRatio/TC_GRDPI_bs16_epoch42_best0.0004506434456512364.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_FixInit/TC_GRDPI_bs16_epoch59_best0.00038361165922143006.pt"
    forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_FixInitNotanh/TC_GRDPI_bs16_epoch7_best0.0004724265052542343.pt"
    # forward_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_Init0_5Notanh/TC_GRDPI_bs16_epoch27_best0.00028589204909729816.pt"

    inverse_PI_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_gamma_InitRandn/TC_GRDPI_bs16_epoch6.pt"
    # inverse_PI_path = "./checkpoints/Sinus_inv_GRDPI_seg50_sr25_relu_InitRandn/TC_GRDPI_bs16_epoch6.pt"

    model_name, model_path, forward = "forward_LSTM_Stair5s", forward_LSTM_path, True
    # model_name, model_path, forward = "forward_GRU", forward_GRU_path, True
    model_name, model_path, forward = "forward_L-bl_Stair5s", forward_LSTMbacklash_path, True
    # model_name, model_path, forward = "inverse_L-bl_40", inverse_LSTMBacklashInv_path, False
    # model_name, model_path, forward = "forward_sum_40", forward_LSTMBacklashSum_path, True
    # model_name, model_path, forward = "inverse_sum", inverse_LSTMBacklashSumInv_path, False
    # model_name, model_path, forward = "forward_bkl", forward_backlash_path, True
    # model_name, model_path, forward = "inverse_bkl", inverse_backlashInv_path, False
    # model_name, model_path, forward = "forward_bl-L_40", forward_backlashLSTM_path, True
    # model_name, model_path, forward = "forward_FNN_40", forward_FNN_path, True
    # model_name, model_path, forward = "inverse_LSTM_40", inverse_LSTM_path, False
    # model_name, model_path, forward = "inverse_FNN", inverse_FNN_path, False
    model_name, model_path, forward = "forward_BoucWen_Sinus", forward_BoucWen_path, True
    # model_name, model_path, forward = "inverse_BoucWen_Sinus", inverse_BoucWen_path, False
    model_name, model_path, forward = "forward_P-I_Sinus", forward_PI_path, True
    # model_name, model_path, forward = "inverse_P-I_Sinus", inverse_PI_path, False

    pos1 = 0
    sr = 25
    seg = 50
    fq, input_dim = False, 1
    act = None
    rm_init = 0
    tip_unit = "deg"
    # folder_path = "./tendon_data/Data_with_Initialization/2024-03-13/all"
    folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/test"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair"
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        motion_name = os.path.basename(path).split(".txt")[0]
        data_path = path
        if "LSTM" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(model_path, data_path, seg=0, forward=forward,
                                                                      fq=fq, input_dim=input_dim, rm_init=rm_init,
                                                                      numl=2, h=64)
        elif "GRU" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_GRU(model_path, data_path, seg=0, forward=forward,
                                                                      fq=fq, input_dim=input_dim, rm_init=rm_init,
                                                                      numl=2, h=64)
        elif "FNN" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(model_path, data_path, seg=seg, forward=forward,
                                                                     fq=fq,
                                                                     input_dim=input_dim, rm_init=rm_init)
        elif "bkl" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_backlash(model_path, data_path, seg=seg,
                                                                          forward=forward, input_dim=input_dim,
                                                                          rm_init=rm_init)
            out_path = "./tendon_data/Data_with_Initialization/BacklashNet/{}.txt".format(motion_name)
            save_backlash(data_path, out_path, time, output)
        elif "L-bl" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_LSTMbacklash(model_path, data_path, seg=seg,
                                                                              forward=forward,
                                                                              input_dim=input_dim, rm_init=rm_init)
        elif "sum" in model_name:
            input_, gt, output, bl_out, time, res_rmse, res_nrmse = test_LSTM_backlash_sum3(model_path, data_path,
                                                                                            seg=seg,
                                                                                            forward=forward,
                                                                                            input_dim=input_dim,
                                                                                            rm_init=rm_init, numl=2,
                                                                                            h=64)
        elif "bl-L" in model_name:
            input_, gt, output, bl_out, time, res_rmse, res_nrmse = test_backlash_LSTM(model_path, data_path, seg=seg,
                                                                                       forward=forward,
                                                                                       input_dim=input_dim,
                                                                                       rm_init=rm_init, numl=2, h=64)
        elif "BoucWen" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_BoucWen(model_path, data_path, seg=seg,
                                                                                       forward=forward,
                                                                                       input_dim=input_dim,
                                                                                       rm_init=rm_init, numl=2, h=64)
        elif "P-I" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_PI(model_path, data_path, seg=seg,
                                                                                       forward=forward,
                                                                                       input_dim=input_dim,
                                                                                       rm_init=rm_init, numl=2, h=64)
            # test_PI_test(model_path, forward=forward)

        # if inverse kinematic model, use forward kinematic model for evaluation
        plot_c = 1
        if not forward:
            pt_path = "./checkpoints/SinusRandom_LSTMbacklash_layer2_seg100_sr25/TC_LSTMbacklash_L2_bs16_epoch185_best2.0394659014840076e-05.pt"
            data_td = output
            data_tip = input_
            forward_output, forward_res_rmse, forward_res_nrmse = Forward_LSTMbacklash(pt_path, data_td, data_tip, input_dim=input_dim, rm_init=0, numl=2, h=64)
            plot_c = 1

        plt.figure(figsize=(88.9 / 25.4 * plot_c, 88.9 / 25.4 * 1.2))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.subplot(2, plot_c, 1)
        plt.tick_params(labelsize=8, pad=0.01, length=2)
        plt.plot(input_, gt, linewidth=0.8, label="Ground truth")
        plt.plot(input_, output, linewidth=0.8, label=model_name + "_model")
        # plt.plot(input_, bl_out, linewidth=0.8, label="Backlash")

        if forward:
            plt.xlabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
            if tip_unit == "deg":
                plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
                plt.title("RMSE:{:.3f} (deg), ME:{:.3f} (deg)".format(rmse_norm(output, gt), max(abs(output - gt))), fontsize=8)
            else:
                plt.ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
                plt.title("RMSE:{:.3f} (mm), ME:{:.3f} (mm)".format(rmse_norm(output, gt), max(abs(output - gt))),
                          fontsize=8)
            plt.xlim([-0.5, 6.5])
            plt.ylim([20, 90])
            # plt.ylim([0, 35])
        else:
            plt.ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
            if tip_unit == "deg":
                plt.xlabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            else:
                plt.xlabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
            plt.title("RMSE:{:.3f} (mm), ME:{:.3f} (mm)".format(rmse_norm(output, gt), max(abs(output - gt))),
                      fontsize=8)
            plt.ylim([-0.5, 6.5])
            plt.xlim([20, 90])
            # plt.ylim([0, 35])
        plt.legend(fontsize=8, frameon=False)

        ax1 = plt.subplot(2, plot_c, plot_c+1)
        ax1.tick_params(labelsize=8, pad=0.01, length=2)
        ax1.plot(time, gt, linewidth=0.8, label="Ground truth")
        ax1.plot(time, output, linewidth=0.8, label=model_name + "_model")
        ax1.set_xlabel("Time (s)", fontsize=8, labelpad=0.01)
        if forward:
            if tip_unit == "deg":
                ax1.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
                ax1.set_ylim([20, 90])
            else:
                ax1.set_ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
                ax1.set_ylim([0, 35])
        else:
            ax1.set_ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
            ax1.set_ylim([-0.1, 6.1])
        plt.legend(fontsize=8, frameon=False)

        # ax2 = ax1.twinx()
        # ax2.tick_params(labelsize=8, pad=0.01, length=2)
        # ax2.plot(time, input_, linewidth=0.8, color='g')
        # ax2.tick_params('y', colors='g')
        #
        # if forward:
        #     ax2.set_ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01, color='g')
        #     ax2.set_ylim([-0.1, 6.1])
        # else:
        #     ax2.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01, color='g')
        #     ax2.set_ylim([0, 90])

        if plot_c==2:
            plt.subplot(2, plot_c, 2)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            plt.plot(gt, input_, 'g', linewidth=0.8, label="Ground truth")
            plt.plot(gt, forward_output, 'r', linewidth=0.8, label="forward-L-bl_model")
            plt.xlabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
            # plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            plt.ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
            plt.title("RMSE:{:.3f} (mm), ME:{:.3f} (mm)".format(rmse_norm(forward_output, input_), max(abs(forward_output - input_))), fontsize=8)
            plt.legend(fontsize=8, frameon=False)

            ax3 = plt.subplot(2, plot_c, 4)
            ax3.tick_params(labelsize=8, pad=0.01, length=2)
            ax3.plot(time, input_, 'g', linewidth=0.8, label="Ground truth")
            ax3.plot(time, forward_output, 'r', linewidth=0.8, label="forward-L-bl_model")
            ax3.set_xlabel("Time (s)", fontsize=8, labelpad=0.01)
            ax3.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            # ax3.set_ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
            ax3.set_ylim([20, 90])
            # ax3.set_ylim([0, 35])
            plt.legend(fontsize=8, frameon=False)

        plt.tight_layout()
        # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.{}Hz_pos{}.jpg".format(test_freq[0], pos1))
        if not os.path.exists("./figures/Data_with_initialization/{}".format(model_name)):
            os.makedirs("./figures/Data_with_initialization/{}".format(model_name))
        plt.savefig("./figures/Data_with_initialization/{}/{}.png".format(model_name, motion_name), dpi=600)
        plt.show()

    acc_file = "./checkpoints/innerTSlackSinus_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_acc_bs16_epoch100.pkl"
    acc = pickle.load(open(acc_file, "rb"))
    plt.figure()
    plt.plot(acc['train'], '-', label='train')
    plt.plot(acc['test'], '-', label='test')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('./results/acc_{}.jpg')
    # plt.show()
