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
# from plot_compare_models import test_FNN
from models.hybrid_model import LSTM_backlash_Net
from models.hybrid_model import BacklashNet
from models.hybrid_model import LSTM_backlash_sum3_Net
from models.hybrid_model import fixedBacklashTipY
from models.hybrid_model import backlash_LSTM_Net
from models.hybrid_model import BacklashInverseNet
from models.hybrid_model import LSTMBacklashInverseNet
from models.hybrid_model import LSTMBacklashSumInvNet
from models.hybrid_model import fixedBacklashTipYInverseNet
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
    X = 7.98 * 25.4 - data[:, 5] + 14  # unit mm
    Y = data[:, 6] - 1.13 * 25.4  # unit mm
    # resample data using fixed sampling rate
    # sample_rate = 100  # Hz
    frequency = round(1 / (time[-1] / 7), 2)
    interp_time = np.arange(10, time[-1], 1 / sample_rate)
    tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
    tip_A_resample = np.interp(interp_time, time, tip_A)
    tip_disp_resample = np.vstack([np.interp(interp_time, time, X), np.interp(interp_time, time, Y)]).T

    # normalization to [0, 1] and pad -1
    tendon_disp = np.hstack([np.ones(seg)*0, tendon_disp_resample / 6]) #6
    tip_A = np.hstack([np.ones(seg)*30/90, tip_A_resample / 90]) #90
    tip_D = np.vstack([np.hstack([np.ones((seg,1))*66/90, np.ones((seg,1))*20/90]), tip_disp_resample / 90])
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
    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    # data[tip_name] = data_filter(data[tip_name])

    joints = data['tendon_disp'][:, np.newaxis, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, 1:2, np.newaxis].astype("float32")
    tips = data[tip_name][:, 1:2, np.newaxis].astype("float32")
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
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    print("weights--------------", weights)
    out = np.array(out)
    if forward:
        out = out/scale
        input = data['tendon_disp'][seg:]*6
        gt = data[tip_name][seg:]*90
        output = out[seg:, 0]*90
    else:
        input = data[tip_name][seg:, 1] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    return input, gt, output, data['time']


def test_LSTM(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    if forward:
        model = LSTMNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl, output_dim=1)
    else:
        model = LSTMNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name][:, 1] = data_filter(data[tip_name][:,1], datatype="disp")

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data[tip_name][:, 1:2].astype("float32") # input tipY

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
    # print(out.shape)

    if forward:
        input = data['tendon_disp'][seg:]*6
        gt = data[tip_name][seg:, 1]*90
        output = out[seg:, 0]*90
    else:
        input = data[tip_name][seg:, 1] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6  # only Y for output
    # print(input.shape)
    # print(gt.shape)
    return input, gt, output, data['time']

def test_GRU(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    if forward:
        model = GRUNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl, output_dim=1)
    else:
        model = GRUNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name][:, 1] = data_filter(data[tip_name][:,1], datatype="disp")

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data[tip_name][:, 1:2].astype("float32") # input tipY

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
        gt = data[tip_name][seg:, 1]*90
        output = out[seg:, 0]*90
    else:
        input = data[tip_name][seg:, 1] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    return input, gt, output, data['time']

def test_FNN(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0):
    device = "cuda"
    if forward:
        model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg, output_dim=1)
    else:
        model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name][:, 1] = data_filter(data[tip_name][:,1], datatype="disp")

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data[tip_name][:, 1:2].astype("float32") # input tipY

    out = []
    for i in range(data['tendon_disp'].shape[0] - seg):
        joint = joints[i + 1:i + seg + 1, 0:input_dim]
        input_ = joint
        output = model(torch.tensor([input_]).to(device))
        predict_pos = output.detach().cpu().numpy()[0]
        out.append(predict_pos[:, 0])
    out = np.array(out)
    # print(out.shape)
    if forward:
        input = data['tendon_disp'][seg:]*6
        gt = data[tip_name][seg:, 1]*90
        output = out[:, 0]*90
    else:
        input = data[tip_name][seg:, 1] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[:, 0] * 6
    return input, gt, output, data['time']

def test_LSTMbacklash(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, rm_init=0, numl=2, h=64):
    device = "cuda"
    if forward:
        model = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl, output_dim=1)
    else:
        model = LSTMBacklashInverseNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    sr = sr
    seg = 0
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name][:, 1] = data_filter(data[tip_name][:, 1], datatype="disp")

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    # pre_pos = data[tip_name][0:1, np.newaxis].astype("float32")
    if forward:
        # pre_pos = data[tip_name][0:1, :].astype("float32")
        # tips = data[tip_name][:, :].astype("float32")
        pre_pos = data[tip_name][0:1, 1:2].astype("float32")  # for tipY only
        tips = data[tip_name][:, 1:2].astype("float32")  # for tipY only
    else:
        pre_pos = data[tip_name][0:1, 1:2].astype("float32")  # for tipY only
        tips = data[tip_name][:, 1:2].astype("float32")  # for tipY only
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
        # out[:, 0] = (1 - out[:, 0]) / 3 + 0.4
        input = data['tendon_disp'][seg:] * 6
        gt = data[tip_name][seg:, 1] * 90
        output = out[seg:, 0] * 90  # output only contain tipY, no tipX
    else:
        input = data[tip_name][seg:, 1] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    return input, gt, output, data['time']

def test_LSTMbacklash_tipDXY(pt_path, data_path, seg=50, sr=25, forward=True, input_dim=1, numl=2, h=64):
    device = "cuda"
    if forward:
        modelX = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl, output_dim=1)
        modelY = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl, output_dim=1)
    sr = sr
    seg = 0
    modelX.load_state_dict(torch.load(pt_path[0], map_location=device))
    modelX.cuda()
    modelX.eval()
    modelY.load_state_dict(torch.load(pt_path[1], map_location=device))
    modelY.cuda()
    modelY.eval()

    tip_name = 'tip_D'
    data = load_data(data_path, seg, sample_rate=sr)
    # data[tip_name] = data_filter(data[tip_name])

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    # pre_pos = data[tip_name][0:1, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, 0:1].astype("float32")
    tips = data[tip_name][:, np.newaxis].astype("float32")
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    outX = []
    hidden = (torch.zeros(modelX.num_layers, 1, modelX.hidden_dim).to(device),
              torch.zeros(modelX.num_layers, 1, modelX.hidden_dim).to(device))
    h_ = hidden
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        tip = tips[i:i + 1, 0:input_dim]
        # pre_pos_ = pre_pos[i-1:i, 0:input_dim]
        if forward:
            output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = modelX(
                torch.tensor([input_]).to(device), torch.tensor([pre_pos]).to(device), h_)
            pre_pos = output.detach().cpu().numpy()[0]
            outX.append(pre_pos[0])

    outY = []
    pre_pos = data[tip_name][0:1, 1:2].astype("float32")
    hidden = (torch.zeros(modelX.num_layers, 1, modelX.hidden_dim).to(device),
              torch.zeros(modelX.num_layers, 1, modelX.hidden_dim).to(device))
    h_ = hidden
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        if forward:
            output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = modelY(
                torch.tensor([input_]).to(device), torch.tensor([pre_pos]).to(device), h_)
            pre_pos = output.detach().cpu().numpy()[0]
            outY.append(pre_pos[0])

    out = np.hstack([outX, outY])
    out[:, 0] = (1 - out[:, 0])/3+0.4
    if forward:
        input = data['tendon_disp'][seg:] * 6
        gt = data[tip_name][seg:] * 90
        output = out[seg:, :] * 90
    else:
        input = data[tip_name][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, :] * 6
    return input, gt, output, data['time']

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
    # plt.plot(filtered*60)
    # plt.show()
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

    tip_name = "tip_D"
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name][:, 1] = data_filter(data[tip_name][:,1], datatype="disp")

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, 1:2].astype("float32")  # for tipY only
    tips = data[tip_name][:, 1:2].astype("float32")  # for tipY only
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
        gt = data[tip_name][seg:, 1] * 90
        output = out[seg:, 0] * 90
    else:
        input = data[tip_name][seg:, 1] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    return input, gt, output, data['time']


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

    if "Sinus12" in pt_path or "StdLoss" in pt_path:
        fixed_backlash = fixedBacklashTipY(datatype="sinus")
    else:
        fixed_backlash = fixedBacklashTipY()
    fixed_backlash.cuda()
    fixed_backlash.eval()

    if "Sinus12" in pt_path or "StdLoss" in pt_path:
        fixed_backlashInv = fixedBacklashTipYInverseNet(datatype="sinus")
    else:
        fixed_backlashInv = fixedBacklashTipYInverseNet()
    fixed_backlashInv.cuda()
    fixed_backlashInv.eval()

    tip_name = "tip_D"
    data = load_data(data_path, seg, sample_rate=sr)
    data[tip_name][:, 1] = data_filter(data[tip_name][:, 1], datatype="disp")

    if forward:
        out_bl = torch.tensor([data[tip_name][0, 1]]).to(device)  # tipY
    else:
        out_bl = torch.tensor([data['tendon_disp'][0]]).to(device)

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, 1:2].astype("float32")  # for tipY only
    tips = data[tip_name][:, 1:2].astype("float32")  # for tipY only
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
        gt = data[tip_name][seg:, 1] * 90
        output = out[seg:, 0] * 90
    else:
        input = data[tip_name][seg:, 1] * 90
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

def inference(model_name, model_path, forward, folder_path):
    sr = 25
    seg = 50
    fq, input_dim, out_dim = False, 1, 2
    act = None
    rm_init = 0
    tip_unit = "mm"

    rmse_me = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        motion_name = os.path.basename(path).split(".txt")[0]
        data_path = path
        if "LSTM" in model_name:
            input_, gt, output, time = test_LSTM(model_path, data_path, seg=0, forward=forward, fq=fq,
                                                 input_dim=input_dim, rm_init=rm_init,
                                                 numl=2, h=64)
        elif "GRU" in model_name:
            input_, gt, output, time = test_GRU(model_path, data_path, seg=0, forward=forward,
                                                fq=fq, input_dim=input_dim, rm_init=rm_init,
                                                numl=2, h=64)
        elif "FNN" in model_name:
            input_, gt, output, time = test_FNN(model_path, data_path, seg=seg, forward=forward, fq=fq,
                                                input_dim=input_dim, rm_init=rm_init)
        elif "FEED" in model_name:
            input_, gt, output, time = test_FNN(model_path, data_path, seg=1, forward=forward, fq=fq, input_dim=input_dim, rm_init=rm_init)

        elif "bkl" in model_name:
            input_, gt, output, time = test_backlash(model_path, data_path, seg=seg, forward=forward, input_dim=input_dim, rm_init=rm_init)
            # out_path = "./tendon_data/Data_with_Initialization/BacklashNet/{}.txt".format(motion_name)
            # save_backlash(data_path, out_path, time, output)
        elif "L_bl" in model_name:
            if len(model_path) != 2:
                input_, gt, output, time = test_LSTMbacklash(model_path, data_path, seg=seg,
                                                             forward=forward,
                                                             input_dim=input_dim, rm_init=rm_init)
            else:
                input_, gt, output, time = test_LSTMbacklash_tipDXY(model_path, data_path, seg=seg, forward=forward,
                                                                    input_dim=input_dim)

        elif "sum" in model_name:
            input_, gt, output, bl_out, time, res_rmse, res_nrmse = test_LSTM_backlash_sum3(model_path, data_path,
                                                                                            seg=seg,
                                                                                            forward=forward,
                                                                                            input_dim=input_dim,
                                                                                            rm_init=rm_init, numl=2,
                                                                                            h=64)
        elif "PI" in model_name:
            input_, gt, output, time = test_PI(model_path, data_path, seg=seg, forward=forward,
                                               input_dim=input_dim,
                                               rm_init=rm_init, numl=2, h=64)
            # test_PI_test(model_path, forward=forward)

        # save rmse and max error
        print(model_name, len(gt.shape), len(output.shape))
        if len(gt.shape)==1:
            # save rmse and max error
            rmse_me.append([rmse_norm(output, gt), max(abs(output - gt))])
        else:
            rmse_me.append([rmse_norm(output[:, 0], gt[:, 0]), rmse_norm(output[:, 1], gt[:, 1]), max(abs(output[:, 0] - gt[:, 0])), max(abs(output[:, 1] - gt[:, 1]))])

        # if inverse kinematic model, use forward kinematic model for evaluation
        plot_c = 1
        if not forward:
            pt_path = "./checkpoints/SinusRandom_LSTMbacklash_layer2_seg100_sr25/TC_LSTMbacklash_L2_bs16_epoch185_best2.0394659014840076e-05.pt"
            data_td = output
            data_tip = input_
            # forward_output, forward_res_rmse, forward_res_nrmse = Forward_LSTMbacklash(pt_path, data_td, data_tip,
            #                                                                            input_dim=input_dim, rm_init=0,
            #                                                                            numl=2, h=64)
            # plot_c = 1

        plt.figure(figsize=(88.9 / 25.4 * plot_c, 88.9 / 25.4 * 2))
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
                plt.title("RMSE:{:.3f} (deg), ME:{:.3f} (deg)".format(rmse_norm(output[:, 0], gt[:, 0]),
                                                                      max(abs(output[:, 0] - gt[:, 0]))), fontsize=8)
            else:
                plt.ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
                # plt.title("RMSE:{:.3f} (mm), ME:{:.3f} (mm)".format(rmse_norm(output[:, 0], gt[:, 0]), max(abs(output[:, 0] - gt[:, 0]))),
                #           fontsize=8)
            plt.xlim([-0.5, 6.5])
            plt.ylim([15, 70])
            # plt.ylim([0, 35])

        plt.legend(fontsize=8, frameon=False)

        ax1 = plt.subplot(2, plot_c, plot_c + 1)
        ax1.tick_params(labelsize=8, pad=0.01, length=2)
        ax1.plot(time, gt, linewidth=0.8, label="Ground truth")
        ax1.plot(time, output, linewidth=0.8, label=model_name + "_model")
        ax1.set_xlabel("Time (s)", fontsize=8, labelpad=0.01)
        if forward:
            if tip_unit == "deg":
                ax1.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
                ax1.set_ylim([15, 70])
            else:
                ax1.set_ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
                ax1.set_ylim([15, 70])
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

        if plot_c == 2:
            plt.subplot(2, plot_c, 2)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            plt.plot(gt, input_, 'g', linewidth=0.8, label="Ground truth")
            plt.plot(gt, forward_output, 'r', linewidth=0.8, label="forward-L-bl_model")
            plt.xlabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
            # plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            plt.ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
            plt.title("RMSE:{:.3f} (mm), ME:{:.3f} (mm)".format(rmse_norm(forward_output, input_),
                                                                max(abs(forward_output - input_))), fontsize=8)
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
        if not os.path.exists("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}".format(model_name)):
            os.makedirs("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}".format(model_name))
        plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}.png".format(model_name, motion_name),
                    dpi=600)
        # plt.show()

    # save rmse and max error to file
    np.save("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name,
                                                                                            folder_path.split("/")[-1]),
            np.array(rmse_me))

if __name__ == '__main__':
    # bkl
    forward_bkl0_path = "./checkpoints/TipY_Sinus12_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch100_best5.017787543692975e-05.pt"
    forward_bkl2_path = "./checkpoints/TipY_Stair5s_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch50_best2.0751378542627208e-05.pt"
    inverse_bkl0_path = "./checkpoints/TipY_Sinus12_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch31_best0.0005138863974328464.pt"
    inverse_bkl2_path = "./checkpoints/TipY_Stair5s_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch38_best0.00013337176838716018.pt"

    # LSTM
    forward_LSTM0_path = "./checkpoints/TipY_Sinus12_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch92_best7.500403701972876e-06.pt"
    forward_LSTM1_path = "./checkpoints/TipD_Sinus_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch98_best1.7129664976922774e-05.pt"
    # forward_LSTM2_path = "./checkpoints/TipD_Stair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch47_best6.750561688022572e-05.pt"
    forward_LSTM2_path = "./checkpoints/TipY_Stair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch91_best5.4858205528164816e-05.pt"
    forward_LSTM4_path = "./checkpoints/TipY_Sinus12_LSTM_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch92_best2.8606445615271998e-05.pt"
    forward_LSTM5_path = "./checkpoints/TipY_Stair5s_LSTM_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch74_best9.731934343497934e-05.pt"
    inverse_LSTM0_path = "./checkpoints/TipY_Sinus12_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch92_best0.0002023994359963884.pt"
    inverse_LSTM2_path = "./checkpoints/TipY_Stair5s_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch94_best0.0007033820387848194.pt"
    inverse_LSTM4_path = "./checkpoints/TipY_Sinus12_LSTMInv_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch100_best0.0007393457975316172.pt"
    inverse_LSTM5_path = "./checkpoints/TipY_Stair5s_LSTMInv_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch85_best0.0012386591517410817.pt"

    # GRU
    forward_GRU0_path = "./checkpoints/TipY_Sinus12_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch95_best7.813755568501316e-06.pt"
    forward_GRU1_path = "./checkpoints/TipD_Sinus_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch88_best1.5408925087362828e-05.pt"
    # forward_GRU2_path = "./checkpoints/TipD_Stair5s_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch83_best6.526875267809373e-05.pt"
    forward_GRU2_path = "./checkpoints/TipY_Stair5s_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch30_best6.314634430011557e-05.pt"
    forward_GRU4_path = "./checkpoints/TipY_Sinus12_GRU_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch89_best2.826980706534717e-05.pt"
    forward_GRU5_path = "./checkpoints/TipY_Stair5s_GRU_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch72_best7.185707403954843e-05.pt"
    # forward_GRU2_path = "./checkpoints/TipY_Stair5s_GRU_layer2_seg50_sr25_rep2/TC_GRU_L2_bs16_epoch90_best5.368311680545698e-05.pt"
    inverse_GRU0_path = "./checkpoints/TipY_Sinus12_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch94_best0.0001957713478380659.pt"
    inverse_GRU2_path = "./checkpoints/TipY_Stair5s_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch72_best0.0006025020751015594.pt"
    inverse_GRU4_path = "./checkpoints/TipY_Sinus12_GRUInv_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch99_best0.0007269885560769277.pt"
    inverse_GRU5_path = "./checkpoints/TipY_Stair5s_GRUInv_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch72_best0.0012006171025513183.pt"

    # feed-forward neural network without history input buffer
    forward_FEED0_path = "./checkpoints/TipY_Sinus12_FNN_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch24_best8.315155785319142e-05.pt"
    forward_FEED2_path = "./checkpoints/TipY_Stair5s_FNN_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch95_best0.00011108379367215093.pt"
    inverse_FEED0_path = "./checkpoints/TipY_Sinus12_FNNInv_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch95_best0.0013392292391720482.pt"
    inverse_FEED2_path = "./checkpoints/TipY_Stair5s_FNNInv_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch23_best0.001457346387906.pt"

    # FNN-HIB
    forward_FNN0_path = "./checkpoints/TipY_Sinus12_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch50_best6.52197912624312e-06.pt"
    forward_FNN1_path = "./checkpoints/TipD_Sinus_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch88_best6.748673614412029e-06.pt"
    # forward_FNN2_path = "./checkpoints/TipD_Stair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch99_best3.745356407885011e-05.pt"
    forward_FNN2_path = "./checkpoints/TipY_Stair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch49_best3.4559697760125834e-05.pt"
    forward_FNN4_path = "./checkpoints/TipY_Sinus12_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch50_best6.52197912624312e-06.pt"
    forward_FNN5_path = "./checkpoints/TipY_Stair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch49_best3.4559697760125834e-05.pt"
    inverse_FNN0_path = "./checkpoints/TipY_Sinus12_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch85_best0.00031741142872003064.pt"
    inverse_FNN2_path = "./checkpoints/TipY_Stair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch74_best0.0004543255959611593.pt"
    inverse_FNN4_path = "./checkpoints/TipY_Sinus12_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch85_best0.00031741142872003064.pt"
    inverse_FNN5_path = "./checkpoints/TipY_Stair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch74_best0.0004543255959611593.pt"

    # LSTMbacklash
    # forward_LSTMbacklash1_path = "./checkpoints/TipD_Sinus_LSTMbacklash_layer2_seg50_sr25/TC"
    forward_L_bl0_path = "./checkpoints/TipY_Sinus12_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch91_best3.400396494157576e-06.pt"
    # forward_L_bl2_path = "./checkpoints/TipD_Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch25_best3.252869031538943e-05.pt"
    forward_L_bl2_path = "./checkpoints/TipY_Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch82_best8.049222554883084e-06.pt"
    forward_L_bl4_path = "./checkpoints/TipY_Sinus12_LSTMBacklash_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch78_best2.3791045350662898e-05.pt"
    forward_L_bl5_path = "./checkpoints/TipY_Stair5s_LSTMBacklash_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch82_best2.2557798844885953e-05.pt"
    inverse_L_bl0_path = "./checkpoints/TipY_Sinus12_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch92_best5.664375203195959e-05.pt"
    inverse_L_bl2_path = "./checkpoints/TipY_Stair5s_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch98_best5.657296034249677e-05.pt"
    inverse_L_bl4_path = "./checkpoints/TipY_Sinus12_LSTMBacklashInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch88_best0.00017873195793072227.pt"
    inverse_L_bl5_path = "./checkpoints/TipY_Stair5s_LSTMBacklashInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch87_best0.0004264048549220971.pt"
    # forward_LSTMbacklash_path = "./checkpoints/TipD_SinusStair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch69_best9.822388446303827e-05.pt"

    # LSTMbacklash tipDX  tipDY
    # forward_LSTMbacklash_path = ["./checkpoints/TipDX_Sinus_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch83_best4.792146729275789e-05.pt",
    #                              "./checkpoints/TipDY_Sinus_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch99_best4.169012954970625e-06.pt"]
    # forward_LSTMbacklash_path = ["./checkpoints/TipDX_Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch37_best5.675998597022651e-05.pt",
    #                              "./checkpoints/TipDY_Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch124_best7.173810630926406e-06.pt"]

    forward_sum0_path = "./checkpoints/TipY_Sinus12_LSTMBacklashsum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch88_best6.190589137607579e-06.pt"
    forward_sum2_path = "./checkpoints/TipY_Stair5s_LSTMBacklashsum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch92_best2.7081706219986598e-05.pt"
    forward_sum4_path = "./checkpoints/TipY_Sinus12_LSTMBacklashsum_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch85_best2.586882070924427e-05.pt"
    forward_sum5_path = "./checkpoints/TipY_Stair5s_LSTMBacklashsum_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch77_best5.316829571263894e-05.pt"
    inverse_sum0_path = "./checkpoints/TipY_Sinus12_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch81_best0.0002328175426858555.pt"
    inverse_sum2_path = "./checkpoints/TipY_Stair5s_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch46_best0.00032137443786413067.pt"
    inverse_sum4_path = "./checkpoints/TipY_Sinus12_LSTMBacklashsumInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch91_best0.0003578359731667054.pt"
    inverse_sum5_path = "./checkpoints/TipY_Stair5s_LSTMBacklashsumInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch33_best0.0005762453273212212.pt"

    folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus"
    folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair10_60s"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair2"

    train_data_type = ["Sinus12", "Sinus", "Stair5s", "SinusStair5s", "Sinus12StdL", "Stair5sStdL"]
    for k in ["forward", "inverse"]:  # , "inverse"   Predict X,Y for forward, use Y as input for inverse
        for m in ["FEED"]:  # "bkl", "PI", "LSTM", "GRU", "FNN", "L_bl", "sum",
            for i in [0, 2]:
                model_path_name = k + "_" + m + str(i) + "_path"
                model_path = globals()[model_path_name]
                forward = True if k == "forward" else False
                model_name = k + "_" + m + "_" + train_data_type[i]
                # model_path = model_path.split("epoch")[0] + "epoch100.pt"
                inference(model_name, model_path, forward, folder_path)
