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
    tip_D = tip_disp_resample / 90
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
        input = data[tip_name][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    gtx = data["tip_D"][seg:, 0] * 90
    gty = data["tip_D"][seg:, 1] * 90
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))

    # inverse kinematics model, use Y as input, map Y to angle and then passed to model prediction
    output_inpY = []
    if not forward:
        data["tip_D"][:, 1] = data_filter(data["tip_D"][:, 1], datatype="disp")
        mapped_tip_A = mapping_Yto_tipA(data["tip_D"][:, 1] * 90) / 90
        tips = mapped_tip_A[:, np.newaxis, np.newaxis].astype("float32")
        joints_ = torch.tensor(tips).to(device)
        out = []
        h_init = joints_[0:1, :, :].expand(1, 1, 4)
        h_ = h_init
        for i in range(data['tendon_disp'].shape[0]):
            input_ = joints_[i:i + 1, 0:input_dim]
            if i == 0:
                input_pre = joints_[i:i + 1, 0:input_dim]
            else:
                input_pre = joints_[i - 1:i, 0:input_dim]
            outp, weights, h_ = model(input_, input_pre, h_)
            pre_pos = outp.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        out = np.array(out)
        output_inpY = out[:, 0] * 6
        # plt.plot(input, output)
        # plt.plot(input, output_inpY)
        # plt.show()

    return input, gt, gtx, gty, output, output_inpY, data['time'], res_rmse, res_nrmse

def test_PI_test(pt_path, forward=True, input_dim=1):
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
    tip_name = 'tip_A'
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

def test_FNN(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, rm_init=0):
    device = "cuda"
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
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
    for i in range(data['tendon_disp'].shape[0] - seg):
        joint = joints[i + 1:i + seg + 1, 0:input_dim]
        input_ = joint
        output = model(torch.tensor([input_]).to(device))
        predict_pos = output.detach().cpu().numpy()[0]
        out.append(predict_pos[0])
    out = np.array(out)
    if forward:
        input = data['tendon_disp'][seg:]*6
        gt = data['tip_A'][seg:]*90
        output = out[:, 0]*90
    else:
        input = data['tip_A'][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[:, 0] * 6
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
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

    tip_name = 'tip_A'
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
        gt = data[tip_name][seg:] * 90
        output = out[seg:, 0] * 90
    else:
        input = data[tip_name][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    gtx = data["tip_D"][seg:, 0] * 90
    gty = data["tip_D"][seg:, 1] * 90
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    return input, gt, gtx, gty, output, data['time'], res_rmse, res_nrmse

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

def Forward_all(model_name, pt_path, data_td, data_tip, input_dim=1, numl=2, h=64, seg=50):
    device = "cuda"
    if "LSTM" in model_name:
        model = LSTMNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    elif "GRU" in model_name:
        model = GRUNet(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    elif "FNN" in model_name:
        model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    elif "L_bl" in model_name:
        model = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
    elif "sum" in model_name:
        model = LSTM_backlash_sum3_Net(inp_dim=input_dim, hidden_dim=h, num_layers=numl)
        fixed_backlash = fixedBacklash()
        fixed_backlash.cuda()
        fixed_backlash.eval()
    elif "bkl" in model_name:
        model = BacklashNet()
    elif "PI" in model_name:
        model = GRDPINet()

    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    joints = data_td[:, np.newaxis].astype("float32")/6
    pre_pos = data_tip[0:1, np.newaxis].astype("float32")/90

    out = []
    if hasattr(model, 'num_layers'):
        hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
                  torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
        h_ = hidden
    if "GRU" in model_name:
        hidden = torch.zeros(model.num_layers, 1, model.hidden_dim).to(device)
        h_ = hidden
    if "PI" in model_name:
        joints = torch.tensor(data_td[:, np.newaxis, np.newaxis].astype("float32") / 6).to(device)
        hidden = joints[0:1, :, :].expand(1, 1, 4)
        h_ = hidden
    out_bl = torch.tensor([data_tip[0]/90]).to(device)
    if "FNN" not in model_name:
        for i in range(0, data_td.shape[0]):
            joint = joints[i:i + 1, 0:input_dim]
            input_ = joint
            if "LSTM" in model_name:
                output, h_ = model(torch.tensor([input_]).to(device), h_)
            elif "GRU" in model_name:
                output, h_ = model(torch.tensor([input_]).to(device), h_)
            elif "L_bl" in model_name:
                output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = model(torch.tensor([input_]).to(device), torch.tensor([pre_pos]).to(device), h_)
            elif "sum" in model_name:
                out_bl = fixed_backlash(torch.tensor([input_]).to(device), out_bl)
                output, h_, out_l = model(torch.tensor([input_]).to(device), out_bl, h_)
            elif "bkl" in model_name:
                output, dynamic_weights, condition_lo, condition_up, condition_mid = model(torch.tensor([joint]).to(device), torch.tensor([pre_pos]).to(device))
            elif "PI" in model_name:
                if i == 0:
                    input_pre = joints[i:i + 1, 0:input_dim]
                else:
                    input_pre = joints[i - 1:i, 0:input_dim]
                output, weights, h_ = model(input_, input_pre, h_)
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        td = data_td
        gt = data_tip
    if "FNN" in model_name:
        out = []
        for i in range(data_td.shape[0] - seg):
            joint = joints[i + 1:i + seg + 1, 0:input_dim]
            input_ = joint
            output = model(torch.tensor([input_]).to(device))
            predict_pos = output.detach().cpu().numpy()[0]
            out.append(predict_pos[0])
        td = data_td[seg:]
        gt = data_tip[seg:]

    out = np.array(out)
    output = out[:, 0] * 90

    res_rmse = rmse_norm(output, gt)
    res_nrmse = res_rmse / (max(gt) - min(gt))
    return td, output, res_rmse, res_nrmse

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

def mapping_Yto_tipA(y):
    coef_y = [-0.00326109, 0.77266619, 0.21515682]
    mapped_tip_A = (-coef_y[1]+np.sqrt(coef_y[1]*coef_y[1]-4*coef_y[0]*(coef_y[2]-y)))/(2*coef_y[0])
    return mapped_tip_A

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
    gtx = data["tip_D"][seg:, 0] * 90
    gty = data["tip_D"][seg:, 1] * 90
    rm_init_number = int(rm_init * len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))

    # inverse kinematics model, use Y as input, map Y to angle and then passed to model prediction
    output_inpY = []
    if not forward:
        data["tip_D"][:, 1] = data_filter(data["tip_D"][:, 1], datatype="disp")
        mapped_tip_A = mapping_Yto_tipA(data["tip_D"][:, 1]*90)/90
        # plt.plot(data['tendon_disp']*6, data["tip_A"])
        # plt.plot(data['tendon_disp']*6, mapped_tip_A)
        # plt.show()
        pre_pos = mapped_tip_A[0:1, np.newaxis].astype("float32")
        tips = mapped_tip_A[:, np.newaxis].astype("float32")
        pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")
        out = []
        for i in range(0, data['tendon_disp'].shape[0]):
            joint = joints[i:i + 1, 0:input_dim]
            tip = tips[i:i + 1, 0:input_dim]
            outp, dynamic_weights = model(torch.tensor([tip]).to(device), torch.tensor([pre_pos]).to(device),
                                            torch.tensor([pre_td]).to(device))  # x_curr, _x_prev, u_pre
            pre_td = outp.detach().cpu().numpy()[0]
            out.append(pre_td[0])
            pre_pos = tip
        out = np.array(out)
        output_inpY = out[:, 0] * 6
        # plt.plot(input, output)
        # plt.plot(input, output_inpY)
        # plt.show()

    return input, gt, gtx, gty, output, output_inpY, data['time'], res_rmse, res_nrmse


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
    # plt.plot(out1[seg:, 0])
    # plt.plot(out2[seg:, 0])
    # plt.show()
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

def inference(model_name, model_path, forward, folder_path, forward_model_path=""):
    print(model_path)
    pos1 = 0
    sr = 25
    seg = 50
    fq, input_dim, out_dim = False, 1, 2
    act = None
    rm_init = 0
    tip_unit = "deg"
    rmse_me = []
    rmse_me_XY = []
    rmse_me_inverse = []
    coef_x = [-3.62136415e-03, -4.61111115e-02, 7.04893736e+01]
    coef_y = [-0.00326109, 0.77266619, 0.21515682]
    p_x = np.poly1d(coef_x)
    p_y = np.poly1d(coef_y)
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        motion_name = os.path.basename(path).split(".txt")[0]
        data_path = path
        if "LSTM" in model_name:
            input_, gt, gtx, gty, output, time, res_rmse, res_nrmse = test_LSTM(model_path, data_path, seg=0,
                                                                                forward=forward,
                                                                                fq=fq, input_dim=input_dim,
                                                                                rm_init=rm_init,
                                                                                numl=2, h=64)
        elif "GRU" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_GRU(model_path, data_path, seg=0, forward=forward,
                                                                     fq=fq, input_dim=input_dim, rm_init=rm_init,
                                                                     numl=2, h=64)
        elif "FNN" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(model_path, data_path, seg=seg, forward=forward,
                                                                     fq=fq,
                                                                     input_dim=input_dim, rm_init=rm_init)
        elif "FEED" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(model_path, data_path, seg=1, forward=forward,
                                                                     fq=fq, input_dim=input_dim, rm_init=rm_init)
        elif "bkl" in model_name:
            input_, gt, gtx, gty, output, output_inpY, time, res_rmse, res_nrmse = test_backlash(model_path, data_path, seg=seg,
                                                                                    forward=forward,
                                                                                    input_dim=input_dim,
                                                                                    rm_init=rm_init)
            # out_path = "./tendon_data/Data_with_Initialization/BacklashNet/{}.txt".format(motion_name)
            # save_backlash(data_path, out_path, time, output)
        elif "L_bl" in model_name:
            input_, gt, gtx, gty, output, time, res_rmse, res_nrmse = test_LSTMbacklash(model_path, data_path, seg=seg,
                                                                                        forward=forward,
                                                                                        input_dim=input_dim,
                                                                                        rm_init=rm_init)
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
        elif "PI" in model_name:
            input_, gt, gtx, gty, output, output_inpY, time, res_rmse, res_nrmse = test_PI(model_path, data_path, seg=seg,
                                                                              forward=forward,
                                                                              input_dim=input_dim,
                                                                              rm_init=rm_init, numl=2, h=64)
            # test_PI_test(model_path, forward=forward)

        # save rmse and max error
        rmse_me.append([rmse_norm(output, gt), max(abs(output - gt))])

        # map tip angle to tip X,Y and save error
        if forward:
            if 'gtx' in locals():
                rmse_me_XY.append(
                    [rmse_norm(p_x(output), gtx), rmse_norm(p_y(output), gty), max(abs(p_x(output) - gtx)),
                     max(abs(p_y(output) - gty))])
        else:
            if 'output_inpY' in locals():
                rmse_me_XY.append([rmse_norm(output_inpY, gt), max(abs(output_inpY - gt))])

        # if inverse kinematic model, use forward kinematic model for evaluation
        plot_c = 1
        if not forward and ("FEED" not in model_name):
            plot_c = 2
            data_td = output
            data_tip = input_
            # pt_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch57_best3.1443552133764724e-05.pt"
            # forward_output, forward_res_rmse, forward_res_nrmse = Forward_LSTMbacklash(pt_path, data_td, data_tip, input_dim=input_dim, rm_init=0, numl=2, h=64)

            forward_gt, forward_output, forward_res_rmse, forward_res_nrmse = Forward_all(model_name, forward_model_path, data_td, data_tip, input_dim=1, numl=2, h=64)


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
                plt.title("RMSE:{:.3f} (deg), ME:{:.3f} (deg)".format(rmse_norm(output, gt), max(abs(output - gt))),
                          fontsize=8)
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

        ax1 = plt.subplot(2, plot_c, plot_c + 1)
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

        if plot_c == 2:
            if "FNN" in model_name:
                input_ = input_[seg:]
                time = time[seg:]
                print(gt.shape, input_.shape)
            plt.subplot(2, plot_c, 2)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            plt.plot(forward_gt, input_, 'g', linewidth=0.8, label="Ground truth")
            plt.plot(forward_gt, forward_output, 'r', linewidth=0.8, label="forward_model")
            plt.xlabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
            plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            # plt.ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)gt, i
            plt.title("RMSE:{:.3f} (deg), ME:{:.3f} (deg)".format(rmse_norm(forward_output, input_),
                                                                max(abs(forward_output - input_))), fontsize=8)
            plt.legend(fontsize=8, frameon=False)
            rmse_me_inverse.append([rmse_norm(forward_output, input_), max(abs(forward_output - input_))])

            ax3 = plt.subplot(2, plot_c, 4)
            ax3.tick_params(labelsize=8, pad=0.01, length=2)
            ax3.plot(time, input_, 'g', linewidth=0.8, label="Ground truth")
            ax3.plot(time, forward_output, 'r', linewidth=0.8, label="forward_model")
            ax3.set_xlabel("Time (s)", fontsize=8, labelpad=0.01)
            ax3.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            # ax3.set_ylabel("Tip disp. (mm)", fontsize=8, labelpad=0.01)
            ax3.set_ylim([20, 90])
            # ax3.set_ylim([0, 35])
            plt.legend(fontsize=8, frameon=False)

        plt.tight_layout()
        # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.{}Hz_pos{}.jpg".format(test_freq[0], pos1))
        if not os.path.exists("./figures/Data_with_initialization_Sinus12Stair5s/{}".format(model_name)):
            os.makedirs("./figures/Data_with_initialization_Sinus12Stair5s/{}".format(model_name))
        plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}.png".format(model_name, motion_name), dpi=600)
        # plt.show()

    # save rmse and max error to file
    np.save("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name, folder_path.split("/")[-1]), np.array(rmse_me))

    if not forward:
        # np.save("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error_L_bl.npy".format(model_name, folder_path.split("/")[-1]), np.array(rmse_me_inverse))
        np.save("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error_forwardSameModel.npy".format(model_name, folder_path.split("/")[-1]), np.array(rmse_me_inverse))

    if True:
        if not os.path.exists("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}_map".format(model_name)):
            os.makedirs("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}_map".format(model_name))
        np.save("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}_map/{}_error.npy".format(model_name,
                                                                                                    folder_path.split(
                                                                                                        "/")[-1]),np.array(rmse_me_XY))


def overplot_models(model_name_list, model_path_name_list, forward, test_file, figure_name):
    pos1 = 0
    sr = 25
    seg = 50
    fq, input_dim, out_dim = False, 1, 2
    act = None
    rm_init = 0
    tip_unit = "deg"
    rmse_me = []
    rmse_me_XY = []
    rmse_me_inverse = []
    coef_x = [-3.62136415e-03, -4.61111115e-02, 7.04893736e+01]
    coef_y = [-0.00326109, 0.77266619, 0.21515682]
    p_x = np.poly1d(coef_x)
    p_y = np.poly1d(coef_y)

    path = test_file
    motion_name = os.path.basename(path).split(".txt")[0]
    data_path = path
    plt.figure(figsize=(88.9 / 25.4/2, 88.9 / 25.4))
    plt.rcParams['font.family'] = 'Times New Roman'
    flag = 0
    for model_name, model_path_name in zip(model_name_list, model_path_name_list):
        model_path = globals()[model_path_name]
        if model_name == "LSTM":
            input_, gt, gtx, gty, output, time, res_rmse, res_nrmse = test_LSTM(model_path, data_path, seg=0,
                                                                                forward=forward,
                                                                                fq=fq, input_dim=input_dim,
                                                                                rm_init=rm_init,
                                                                                numl=2, h=64)
        elif "GRU" in model_name:
            input_, gt, output, time, res_rmse, res_nrmse = test_GRU(model_path, data_path, seg=0, forward=forward,
                                                                     fq=fq, input_dim=input_dim, rm_init=rm_init,
                                                                     numl=2, h=64)
        elif model_name == "FNN-HIB":
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(model_path, data_path, seg=seg, forward=forward,
                                                                     fq=fq,
                                                                     input_dim=input_dim, rm_init=rm_init)
        elif model_name == "FNN":
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(model_path, data_path, seg=1, forward=forward,
                                                                     fq=fq,
                                                                     input_dim=input_dim, rm_init=rm_init)
        elif model_name == "Backlash":
            input_, gt, gtx, gty, output, output_inpY, time, res_rmse, res_nrmse = test_backlash(model_path, data_path, seg=seg,
                                                                                    forward=forward,
                                                                                    input_dim=input_dim,
                                                                                    rm_init=rm_init)
        elif "HB1" in model_name:
            input_, gt, gtx, gty, output, time, res_rmse, res_nrmse = test_LSTMbacklash(model_path, data_path, seg=seg,
                                                                                        forward=forward,
                                                                                        input_dim=input_dim,
                                                                                        rm_init=rm_init)
        elif "HB2" in model_name:
            input_, gt, output, bl_out, time, res_rmse, res_nrmse = test_LSTM_backlash_sum3(model_path, data_path,
                                                                                            seg=seg,
                                                                                            forward=forward,
                                                                                            input_dim=input_dim,
                                                                                            rm_init=rm_init, numl=2,
                                                                                            h=64)
        elif "PI" in model_name:
            input_, gt, gtx, gty, output, output_inpY, time, res_rmse, res_nrmse = test_PI(model_path, data_path, seg=seg,
                                                                              forward=forward,
                                                                              input_dim=input_dim,
                                                                              rm_init=rm_init, numl=2, h=64)

        # save rmse and max error
        rmse_me.append([rmse_norm(output, gt), max(abs(output - gt))])

        print(len(time))
        ind1, ind2 = 1200, 1500
        # ind1, ind2 = 100, 300
        # input_, gt, time, output = input_[ind1:ind2], gt[ind1:ind2], time[ind1:ind2], output[ind1:ind2]

        plt.subplot(2, 1, 2)
        plt.tick_params(labelsize=7, pad=0.01, length=2)
        if flag == 0:
        #     plt.plot(input_, gt, linewidth=0.8, label="Experimental data")
            plt.plot(10, 10)
        # plt.plot(input_, output, linewidth=0.8, label=model_name)
        # plt.plot(input_, bl_out, linewidth=0.8, label="Backlash")
        plt.plot(time-10, output - gt, linewidth=0.8, label=model_name)
        if model_name == "FNN-HIB":
            cl = "orange"
        elif model_name == "FNN":
            cl = "r"
        else:
            cl = "g"
        # plt.axhline(y=np.mean(output-gt), linewidth=0.8, color=cl, linestyle="--")

        if forward:
            # plt.xlabel("Tendon displacement (mm)", fontsize=8, labelpad=0.01)
            plt.xlabel("t (s)", fontsize=7, labelpad=0.01)
            if tip_unit == "deg":
                plt.ylabel("Tip angle error (degrees)", fontsize=7, labelpad=0.01)
                # plt.title("RMS Error={:.3f} (deg), Maximum error={:.3f} (deg)".format(rmse_norm(output, gt), max(abs(output - gt))), fontsize=8)
            else:
                plt.ylabel("Tip displacement (mm)", fontsize=7, labelpad=0.01)
                # plt.title("RMS Error={:.3f} (mm), Maximum error={:.3f} (mm)".format(rmse_norm(output, gt), max(abs(output - gt))), fontsize=8)
            # plt.xlim([-0.5, 6.5])
            # plt.xlim([-1, 65])
            plt.ylim([-6, 6])
            # plt.ylim([0, 35])
        else:
            plt.ylabel("Tendon displacement error (mm)", fontsize=7, labelpad=0.01)
            plt.xlabel("t (s)", fontsize=7, labelpad=0.01)
            # if tip_unit == "deg":
            #     plt.xlabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
            # else:
            #     plt.xlabel("Tip displacement (mm)", fontsize=8, labelpad=0.01)
            # plt.title("RMS Error={:.3f} (mm), Maximum error={:.3f} (mm)".format(rmse_norm(output, gt), max(abs(output - gt))), fontsize=8)
            # plt.ylim([-0.5, 6.5])
            # plt.xlim([-2, 66])
            plt.ylim([-0.6, 0.6])
        plt.legend(fontsize=7, frameon=False, ncol=3, handlelength=1)
        plt.grid(True, linestyle=':')

        ax1 = plt.subplot(2, 1, 1)
        ax1.tick_params(labelsize=7, pad=0.01, length=2)
        if flag == 0:
            ax1.plot(time-10, gt, linewidth=0.8, label="Experimental data")
        ax1.plot(time-10, output, linewidth=0.8, label=model_name)
        # ax1.plot(time, output, linewidth=0.8, label=model_name+" (RMS Error={:.2f}, Maximum Error={:.2f})".format(rmse_me[-1][0],rmse_me[-1][1]))
        ax1.set_xlabel("t (s)", fontsize=7, labelpad=0.01)
        # ax1.set_xlim([-2, 66])
        if forward:
            if tip_unit == "deg":
                ax1.set_ylabel("Tip angle (degrees)", fontsize=7, labelpad=0.01)
                ax1.set_ylim([20, 90])
            else:
                ax1.set_ylabel("Tip displacement (mm)", fontsize=7, labelpad=0.01)
                ax1.set_ylim([0, 35])
        else:
            ax1.set_ylabel("Tendon displacement (mm)", fontsize=7, labelpad=0.01)
            ax1.set_ylim([-0.2, 6.2])
        flag = 1
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=7, frameon=False, ncol=2, handlelength=1)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.18, bottom=0.08, left=0.17, right=0.98)
    # plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/overplot_models/{}_{}_{}.png".format(figure_name, forward, motion_name), dpi=600)
    plt.savefig("./figures/RAL_plots/{}_{}_{}.svg".format(figure_name, forward, motion_name))
    # plt.savefig("./figures/RAL_plots/{}_{}_{}.png".format(figure_name, forward, motion_name), dpi=600)
    # plt.show()

if __name__ == '__main__':
    # backlash
    forward_bkl0_path = "./checkpoints/Sinus12_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch43_best7.098719712909467e-05.pt"
    forward_bkl1_path = "./checkpoints/Sinus_BacklashNet_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch73_best8.678210576625256e-05.pt"
    forward_bkl2_path = "./checkpoints/Stair5s_BacklashNet_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch167_best4.843862231306654e-05.pt"
    forward_bkl3_path = "./checkpoints/SinusStair5s_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch192_best9.588336713534795e-05.pt"
    forward_bkl4_path = "./checkpoints/Sinus12_Backlash_layer2_seg50_sr25_StdLoss/TC_Backlash_L2_bs16_epoch32_best0.00011902969830164996.pt"
    forward_bkl5_path = "./checkpoints/Stair5s_Backlash_layer2_seg50_sr25_StdLoss/TC_Backlash_L2_bs16_epoch26_best8.545389504260605e-05.pt"
    inverse_bkl0_path = "./checkpoints/Sinus12_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch49_best0.00031211362026321393.pt"
    inverse_bkl1_path = "./checkpoints/Sinus_BacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch236_best0.00036487386042007513.pt"
    inverse_bkl2_path = "./checkpoints/Stair5s_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch178_best0.00021264683127810712.pt"
    inverse_bkl3_path = "./checkpoints/SinusStair5s_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch256_best0.0003787306132002444.pt"
    inverse_bkl4_path = "./checkpoints/Sinus12_BacklashInv_layer2_seg50_sr25_StdLoss/TC_Backlash_L2_bs16_epoch65_best0.00042952897638315335.pt"
    inverse_bkl5_path = "./checkpoints/Stair5s_BacklashInv_layer2_seg50_sr25_StdLoss/TC_Backlash_L2_bs16_epoch59_best0.0012970864532574537.pt"

    # LSTM
    forward_LSTM0_path = "./checkpoints/Sinus12_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch88_best6.226386176422238e-05.pt"
    forward_LSTM1_path = "./checkpoints/Sinus_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch80_best7.461752777869281e-05.pt"
    forward_LSTM2_path = "./checkpoints/Stair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch89_best0.0003633003450280133.pt"
    forward_LSTM4_path = "./checkpoints/Sinus12_LSTM_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch63_best0.00011929163732323407.pt"
    forward_LSTM5_path = "./checkpoints/Stair5s_LSTM_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch94_best0.0004555917950624245.pt"
    forward_LSTM3_path = "./checkpoints/SinusStair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch108_best0.0002405683620917971.pt"
    inverse_LSTM0_path = "./checkpoints/Sinus12_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch67_best0.00020092543066615084.pt"
    inverse_LSTM1_path = "./checkpoints/Sinus_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch99_best0.00019427379040773643.pt"
    inverse_LSTM2_path = "./checkpoints/Stair5s_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch97_best0.0007411067635985091.pt"
    inverse_LSTM3_path = "./checkpoints/SinusStair5s_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch95_best0.0004180659107842381.pt"
    inverse_LSTM4_path = "./checkpoints/Sinus12_LSTMInv_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch98_best0.0004227304646822934.pt"
    inverse_LSTM5_path = "./checkpoints/Stair5s_LSTMInv_layer2_seg50_sr25_StdLoss/TC_LSTM_L2_bs16_epoch89_best0.0010909832319215176.pt"

    # GRU
    forward_GRU0_path = "./checkpoints/Sinus12_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch95_best6.017432770022424e-05.pt"
    forward_GRU1_path = "./checkpoints/Sinus_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch97_best7.328718462396587e-05.pt"
    forward_GRU2_path = "./checkpoints/Stair5s_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch99_best0.0003257222237152746.pt"
    forward_GRU3_path = "./checkpoints/SinusStair5s_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch49_best0.0002430327041730081.pt"
    forward_GRU4_path = "./checkpoints/Sinus12_GRU_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch89_best0.00011463792482876063.pt"
    forward_GRU5_path = "./checkpoints/Stair5s_GRU_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch95_best0.0003830992112246652.pt"
    inverse_GRU0_path = "./checkpoints/Sinus12_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch95_best0.00017894160070378953.pt"
    inverse_GRU1_path = "./checkpoints/Sinus_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch91_best0.0001758793646408545.pt"
    inverse_GRU2_path = "./checkpoints/Stair5s_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch92_best0.0005654142509835462.pt"
    inverse_GRU3_path = "./checkpoints/SinusStair5s_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch92_best0.00036274320216059.pt"
    inverse_GRU4_path = "./checkpoints/Sinus12_GRUInv_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch86_best0.0004130941730788133.pt"
    inverse_GRU5_path = "./checkpoints/Stair5s_GRUInv_layer2_seg50_sr25_StdLoss/TC_GRU_L2_bs16_epoch46_best0.0010671035621300459.pt"

    # feed-forward neural network without history input buffer
    forward_FEED0_path = "./checkpoints/Sinus12_FNN_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch99_best0.0006063718306408687.pt"
    forward_FEED2_path = "./checkpoints/Stair5s_FNN_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch87_best0.0006989452289417386.pt"
    inverse_FEED0_path = "./checkpoints/Sinus12_FNNInv_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch99_best0.0015899114924567666.pt"
    inverse_FEED2_path = "./checkpoints/Stair5s_FNNInv_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch23_best0.0017457388438439617.pt"

    # FNN-HIB
    forward_FNN0_path = "./checkpoints/Sinus12_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch87_best6.51515959665024e-05.pt"
    forward_FNN1_path = "./checkpoints/Sinus_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch92_best2.9046731570916665e-05.pt"
    forward_FNN2_path = "./checkpoints/Stair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch91_best0.00019324814696567103.pt"
    forward_FNN3_path = "./checkpoints/SinusStair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch81_best0.00014734179199472742.pt"
    forward_FNN4_path = "./checkpoints/Sinus12_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch87_best6.51515959665024e-05.pt"    # no std loss
    forward_FNN5_path = "./checkpoints/Stair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch91_best0.00019324814696567103.pt"  # no std loss
    inverse_FNN0_path = "./checkpoints/Sinus12_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch96_best0.0003058897442921686.pt"
    inverse_FNN1_path = "./checkpoints/Sinus_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch74_best0.0002357431442919372.pt"
    inverse_FNN2_path = "./checkpoints/Stair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch90_best0.0004392276931044069.pt"
    inverse_FNN3_path = "./checkpoints/SinusStair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch74_best0.00046465756479717257.pt"
    inverse_FNN4_path = "./checkpoints/Sinus12_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch96_best0.0003058897442921686.pt"  # no std loss
    inverse_FNN5_path = "./checkpoints/Stair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch90_best0.0004392276931044069.pt"   # no std loss

    # LSTMbacklash
    forward_L_bl0_path = "./checkpoints/Sinus12_LSTMbacklash_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch87_best2.454245024334038e-05.pt"
    forward_L_bl1_path = "./checkpoints/Sinus_LSTMbacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch85_best2.3004282411420718e-05.pt"
    forward_L_bl2_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch57_best3.1443552133764724e-05.pt"
    forward_L_bl3_path = "./checkpoints/SinusStair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch71_best4.309366758046535e-05.pt"
    forward_L_bl4_path = "./checkpoints/Sinus12_LSTMBacklash_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch85_best7.111722879926674e-05.pt"
    forward_L_bl5_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch96_best6.719550895338346e-05.pt"
    forward_L_bl6_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25_b4/TC_LSTMBacklash_L2_bs4_epoch100.pt"
    forward_L_bl6_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25_b2/TC_LSTMBacklash_L2_bs2_epoch25_best3.058737506998178e-05.pt"

    inverse_L_bl0_path = "./checkpoints/Sinus12_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMBacklash_L2_bs16_epoch89_best0.00016522284143623742.pt"
    inverse_L_bl1_path = "./checkpoints/Sinus_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch98_best0.00017395542916778665.pt"
    inverse_L_bl2_path = "./checkpoints/Stair5s_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch99_best0.00014391199280604876.pt"
    inverse_L_bl3_path = "./checkpoints/SinusStair5s_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch86_best0.0002507692252573393.pt"
    inverse_L_bl4_path = "./checkpoints/Sinus12_LSTMBacklashInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch100_best0.0002931389632673624.pt"
    inverse_L_bl5_path = "./checkpoints/Stair5s_LSTMBacklashInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklash_L2_bs16_epoch80_best0.0012929419171996415.pt"

    # LSTMBacklashSum
    forward_sum0_path = "./checkpoints/Sinus12_LSTMBacklashsum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch71_best2.922072872024728e-05.pt"
    forward_sum1_path = "./checkpoints/Sinus_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch79_best2.8625850279502753e-05.pt"
    forward_sum2_path = "./checkpoints/Stair5s_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch59_best7.008730863162782e-05.pt"
    forward_sum3_path = "./checkpoints/SinusStair5s_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch86_best7.081233312771069e-05.pt"
    forward_sum4_path = "./checkpoints/Sinus12_LSTMBacklashsum_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch71_best7.803346108465286e-05.pt"
    forward_sum5_path = "./checkpoints/Stair5s_LSTMBacklashsum_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch32_best0.00012618057347329643.pt"
    inverse_sum0_path = "./checkpoints/Sinus12_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch68_best0.0001435664096127518.pt"
    inverse_sum1_path = "./checkpoints/Sinus_LSTMBacklashSumInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch89_best0.00012101194886905062.pt"
    inverse_sum2_path = "./checkpoints/Stair5s_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch36_best0.0001648359628840505.pt"
    inverse_sum3_path = "./checkpoints/SinusStair5s_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch43_best0.0002371085314383386.pt"
    inverse_sum4_path = "./checkpoints/Sinus12_LSTMBacklashsumInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch96_best0.00022921742956069383.pt"
    inverse_sum5_path = "./checkpoints/Stair5s_LSTMBacklashsumInv_layer2_seg50_sr25_StdLoss/TC_LSTMBacklashsum_L2_bs16_epoch40_best0.0003316304117712813.pt"

    # rate-dependent P-I model
    forward_PI0_path = "./checkpoints/Sinus12_GRDPI_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch39_best0.0002609164099946308.pt"
    forward_PI1_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch111_best0.0002733594585028556.pt"
    forward_PI2_path = "./checkpoints/Stair5s_GRDPI_seg50_sr25_relu_noGamma_a4_new/TC_GRDPI_bs16_epoch88_best0.00047542826902668273.pt"
    forward_PI3_path = "./checkpoints/SinusStair5s_GRDPI_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch106_best0.0004355512255413395.pt"
    forward_PI4_path = "./checkpoints/Sinus12_GRDPI_seg50_sr25_relu_noGamma_a4_StdLoss/TC_GRDPI_bs16_epoch17_best0.00031510412554780487.pt"
    forward_PI5_path = "./checkpoints/Stair5s_GRDPI_seg50_sr25_relu_noGamma_a4_StdLoss/TC_GRDPI_bs16_epoch75_best0.0005143211370552077.pt"
    inverse_PI0_path = "./checkpoints/Sinus12_GRDPIInv_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch51_best0.0003881157275221388.pt"
    inverse_PI1_path = "./checkpoints/Sinus_GRDPIInv_seg50_sr25_relu_inverseForm_a4/TC_GRDPI_bs16_epoch68_best0.00029903930667156387.pt"
    inverse_PI2_path = "./checkpoints/Stair5s_GRDPIInv_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch42_best0.0003211082138780815.pt"
    inverse_PI3_path = "./checkpoints/SinusStair5s_GRDPIInv_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch111_best0.0004364665205876085.pt"
    inverse_PI4_path = "./checkpoints/Sinus12_GRDPIInv_seg50_sr25_relu_noGamma_a4_StdLoss/TC_GRDPI_bs16_epoch84_best0.0005032346028504738.pt"
    inverse_PI5_path = "./checkpoints/Stair5s_GRDPIInv_seg50_sr25_relu_noGamma_a4_StdLoss/TC_GRDPI_bs16_epoch11_best0.0013450650407725738.pt"

    folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus"
    folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair10_60s"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair"
    # folder_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair2"

    train_data_type = ["Sinus12", "Sinus", "Stair5s", "SinusStair5s", "Sinus12StdL", "Stair5sStdL", "Stair5sb2"]
    for k in ["forward"]: # ,"forward", "inverse"
        for m in ["L_bl"]: #"bkl", "LSTM", "GRU", "FNN", "L_bl", "sum",
            for i in [6]:
                model_path_name = k+"_"+m+str(i)+"_path"
                model_path = globals()[model_path_name]
                forward = True if k == "forward" else False
                model_name = k+"_"+m+"_"+train_data_type[i]
                # model_path = model_path.split("epoch")[0] + "epoch100.pt"
                # inference(model_name, model_path, forward, folder_path, forward_model_path=globals()["forward_"+m+str(i)+"_path"])


    test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/random_stair1-10_11.txt"
    # test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/random_stair1-10_6.txt"
    # test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/random_stair1-10_19.txt"
    model_path_name_list = ["forward_bkl2_path", "forward_LSTM5_path", "forward_L_bl5_path"]
    model_name_list = ["Backlash", "LSTM", "LSTM-Backlash Serial Hybrid"]
    # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="Forward_stair")
    model_path_name_list = ["inverse_bkl5_path", "inverse_LSTM5_path", "inverse_L_bl5_path"]
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Inverse_stair")
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Inverse_stair_segment")

    test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus/MidBL_0.45Hz.txt"
    model_path_name_list = ["forward_bkl0_path", "forward_LSTM0_path", "forward_L_bl0_path"]
    # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="Forward_sinus")
    model_path_name_list = ["inverse_bkl4_path", "inverse_LSTM0_path", "inverse_L_bl0_path"]
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Inverse_sinus")
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Inverse_sinus_segment")

    test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus/MidBL_0.45Hz.txt"
    model_name_list = ["PI", "LSTM", "FNN-HIB"]
    model_path_name_list = ["forward_PI4_path", "forward_LSTM0_path", "forward_FNN0_path"]
    # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="PI_LSTM_FNN_Forward_sinus")
    model_path_name_list = ["inverse_PI4_path", "inverse_LSTM0_path", "inverse_FNN0_path"]
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="PI_LSTM_FNN_Inverse_sinus")
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Inverse_sinus_segment")

    test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/random_stair1-10_11.txt"
    model_path_name_list = ["forward_bkl2_path", "forward_LSTM5_path", "forward_FNN2_path"]
    model_name_list = ["Backlash", "LSTM", "FNN-HIB"]
    # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="Backlash_LSTM_FNN_Forward_stair")
    model_path_name_list = ["inverse_bkl5_path", "inverse_LSTM5_path", "inverse_FNN2_path"]
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Backlash_LSTM_FNN_Inverse_stair")
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Inverse_stair_segment")

    # test_file = "./tendon_data/Data_with_Initialization/SinusStair5s/sinus/MidBL_0.45Hz.txt"
    # model_name_list = ["PI", "LSTM", "FNN"]
    # model_path_name_list = ["forward_PI4_path", "forward_LSTM0_path", "forward_FEED0_path"]
    # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="PI_LSTM_Feed_Forward_sinus")
    # model_path_name_list = ["inverse_PI4_path", "inverse_LSTM0_path", "inverse_FEED0_path"]
    # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="PI_LSTM_Feed_Inverse_sinus")

    # test_path = "./tendon_data/Data_with_Initialization/Stair5s/train"
    test_path = "./tendon_data/Data_with_Initialization/SinusStair5s/stair1_10s/"
    # for test_name in os.listdir(test_path):
    for test_name in ["random_stair1-10_1.txt"]:
        test_file = os.path.join(test_path, test_name)
        # model_path_name_list = ["forward_bkl2_path", "forward_LSTM5_path", "forward_FEED2_path"]
        # model_name_list = ["Backlash", "LSTM", "FNN"]
        # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="Backlash_LSTM_Feed0_Forward_stairv2")
        # model_path_name_list = ["inverse_bkl5_path", "inverse_LSTM5_path", "inverse_FEED2_path"]
        # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="Backlash_LSTM_Feed0_Inverse_stairv2")

        # model_path_name_list = ["forward_PI2_path", "forward_sum5_path", "forward_FEED2_path"]
        # model_name_list = ["PI", "HB2", "FNN"]
        # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="PI_HB2_Feed2_Forward_stairv2")
        # model_path_name_list = ["inverse_PI2_path", "inverse_sum2_path", "inverse_FEED2_path"]
        # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="PI_HB2_Feed2_Inverse_stairv2")

        # model_path_name_list = ["forward_PI2_path", "forward_L_bl5_path", "forward_FEED2_path"]
        # model_name_list = ["PI", "HB1", "FNN"]
        # overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="PI_HB1_Feed2_Forward_stairv2")
        # model_path_name_list = ["inverse_PI2_path", "inverse_L_bl5_path", "inverse_FEED2_path"]
        # overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="PI_HB1_Feed2_Inverse_stairv2")

        model_path_name_list = ["forward_FNN2_path", "forward_sum5_path", "forward_FEED2_path"]
        model_name_list = ["FNN-HIB", "HB2", "FNN"]
        overplot_models(model_name_list, model_path_name_list, True, test_file, figure_name="FNN-HIB_HB2_Feed2_Forward_stairv2")
        model_path_name_list = ["inverse_FNN2_path", "inverse_sum2_path", "inverse_FEED2_path"]
        overplot_models(model_name_list, model_path_name_list, False, test_file, figure_name="FNN-HIB_HB2_Feed2_Inverse_stairv2")



