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

def test_LSTM(pt_path, data_path, forward=True, fq=False, input_dim=1):
    device = "cuda"
    model = LSTMNet(inp_dim=input_dim, num_layers=2)
    sr = 25
    seg = 50
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    data = load_data(data_path, seg, sample_rate=sr)
    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
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
        gt = data['tip_A'][seg:]*90
        output = out[seg:, 0]*90
    else:
        input = data['tip_A'][seg:] * 90
        gt = data['tendon_disp'][seg:] * 6
        output = out[seg:, 0] * 6
    res_rmse = rmse_norm(output, gt)
    res_nrmse = res_rmse/(max(gt)-min(gt))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def compute_gradient(model, input_array, hidden, seg=50):
    # hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim),
    #           torch.zeros(model.num_layers, 1, model.hidden_dim))
    # input_tensor = torch.randn(1, seg, requires_grad=True)
    # input_tensor = torch.ones(1, seg, requires_grad=True)
    input_tensor = torch.tensor(input_array, requires_grad=True)
    output, _ = model(input_tensor, hidden)
    # Calculate the contribution of each input (gradient of the loss with respect to the input)
    contributions = grad(output[0, -1, 0], input_tensor)[0]
    x = contributions.detach().cpu().numpy()[0]
    # print(x.shape)
    return x

if __name__ == '__main__':
    path = "./checkpoints/TC_LSTM_45ca_seg50_lossall/TP_LSTM_L2_bs16_epoch168_best0.0003329007666325197.pt"
    path = "./checkpoints/test/TC_LSTM_45ca_fakedata_seg50_lossall_sr100/TP_LSTM_L2_bs16_epoch500.pt"
    path = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3_loss1/TC_LSTM_L2_bs16_epoch76_best0.0005161732333164995.pt"
    path = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3_loss10000/TC_LSTM_L2_bs16_epoch100_best1.896596786374947.pt"
    # path = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr50_loss1_epoch1000/TC_LSTM_L2_bs16_epoch900.pt"
    # path = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr100_loss1_epoch1000/TC_LSTM_L2_bs16_epoch459_best0.0004922032621111817.pt"

    # path = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_LSTM_L2_bs16_epoch637_best0.0002622583022068299.pt"
    # path = "./checkpoints/Inverse_TC_LSTM_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_LSTM_L2_bs16_epoch755_best0.00045527471983977766.pt"

    path = "./checkpoints/TC_LSTM_45ca_1rep_seg50_sr100_rsFreq_loss1_epoch500/TC_LSTM_L2_bs16_epoch234_best0.0003109167695504806.pt"   # not too bad, not good
    # path = "./checkpoints/TC_LSTM_45ca_1rep_seg50_sr100_rsFFreqT_loss1_epoch500/TC_LSTM_L2_bs16_epoch193_best0.0005387310118590983.pt"  # too bad
    # path = "./checkpoints/TC_LSTM_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch273_best1.428707904951728.pt"  #  good
    # path = "./checkpoints/TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch258_best1.2960718440026353.pt"   # not bad, bad when freq is low

    # fq, input_dim = False, 1
    path = "../hysteresis_ISMR/checkpoints/FixValidateData_TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch248_best1.3202891629189253.pt"
    # path = "./checkpoints/FixValidateData_Inverse_TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch352_best2.2775139451026916.pt"

    # fq, input_dim = True, 2
    # path = "../hysteresis_ISMR/checkpoints/FixValidateData_TC_LSTM_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch425_best1.3930550070564731.pt"
    # path = "./checkpoints/FixValidateData_Inverse_TC_LSTM_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch363_best2.370356702337078.pt"

    # path = "./checkpoints/LSTM_45ca1rep_seg50_sr25_NOrsNOfq_loss1_NOwd/TC_LSTM_L2_bs16_epoch371_best0.00012022016062473995.pt"
    # path = "./checkpoints/LSTM_45ca4rep_seg50_sr25_NOrsNOfq_NOwd/TC_LSTM_L2_bs16_epoch100_best0.00010517176146020142.pt"

    path = "./checkpoints/LSTM_Train135Hz1rep_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch446_best0.00010628091521853815.pt"
    # path = "./checkpoints/LSTM_Train135Hz3rep_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch430_best7.390102776035093e-05.pt"
    path = "./checkpoints/LSTM_Train135Hz1rep_seg50_sr25_NOrsNOfq_TCdataset/TC_LSTM_L2_bs16_epoch426_best0.00012053733629829503.pt"
    path = "./checkpoints/LSTM_Train135Hz3rep_seg50_sr25_NOrsNOfq_TCdataset/TC_LSTM_L2_bs16_epoch495_best8.714309888516518e-05.pt"

    #  different window size
    path = "./checkpoints/Train12345Hz1rep_LSTM_seg2_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch385_best0.0009705254304532141.pt"
    path = "./checkpoints/Train12345Hz1rep_LSTM_seg10_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch429_best0.0004312794906606733.pt"
    # path = "./checkpoints/Train12345Hz1rep_LSTM_seg20_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch494_best0.00024159783231807034.pt"
    # path = "./checkpoints/Train12345Hz1rep_LSTM_seg30_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch483_best0.000176327922231123.pt"
    # path = "./checkpoints/Train12345Hz1rep_LSTM_seg40_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch495_best0.00013250087551185921.pt"
    path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch488_best0.00010981084632440833.pt"
    path = "./checkpoints/Train12345Hz1rep_LSTM_seg100_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch451_best6.0820018704244984e-05.pt"

    #  different sampling rate
    # path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr10_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch449_best8.010477757858071e-05.pt"
    path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch488_best0.00010981084632440833.pt"
    # path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr50_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch468_best0.00016143671042987854.pt"
    # path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr100_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch377_best0.0002389827078441652.pt"

    # path = "./checkpoints/Train12345Hz1rep_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch477_best0.0001874904317139533.pt"
    # path = "./checkpoints/Train12345Hz1rep_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd_pos1/TC_LSTM_L2_bs16_epoch399_best0.00014465532957102993.pt"

    forward = True
    pos1 = 0
    sr = 25
    fq, input_dim = False, 1
    act = None
    # fq, input_dim = True, 2
    # load trained model
    device = "cpu"
    model = LSTMNet(inp_dim=input_dim, num_layers=2, act=act)
    model.load_state_dict(torch.load(path, map_location=device))
    # model.cuda()
    model.eval()

    seg = 50
    # load data
    path = "./tendon_data/45ca_1rep/test/MidBL_015Hz_repn.txt"
    # path = "./tendon_data/Train135Hz_1rep/train/0BL_01hz_rep2.txt"
    # path = "./tendon_data/Train135Hz_1rep/validate/EndBL_01hz_rep5.txt"
    # path = "./tendon_data/Train135Hz_1rep/validate/0BL_03hz_rep4.txt"
    # path = "./tendon_data/Train135Hz_1rep/validate/0BL_05Hz_rep4.txt"
    # path = "./tendon_data/Train135Hz_1rep/test/0BL_02Hz_rep1.txt"
    # path = "./tendon_data/Train135Hz_1rep/test/0BL_04hz_rep2.txt"
    data_path = os.path.join(path)
    data = load_data(data_path, seg, sample_rate=sr)

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
        pos = data['tip_A'][:, np.newaxis].astype("float32")
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
        pos = data['tendon_disp'][:, np.newaxis].astype("float32")
    if fq:
        freq = data['freq'][:, np.newaxis].astype("float32")
        joints = np.concatenate([joints, freq], axis=1)

    pre_pos = np.array([[-1.0]]).astype("float32")

    out = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    torch.cuda.synchronize()
    start_time = time.time()
    t = 0
    cg = 0
    for i in range(data['tendon_disp'].shape[0]):
        # if i >= seg and i < data['tendon_disp'].shape[0]-seg:
        if i == 100 and pos1 == 0:
            contribute = (compute_gradient(model, np.array([joints[i:i + seg, 0:input_dim]]), hidden, seg=seg))
            cg += np.abs(contribute)
        joint = joints[i:i + 1, 0:input_dim]
        if pos1 == 1:
            input_ = np.hstack([joint, pre_pos])[np.newaxis, :]  # freq
        else:
            input_ = np.array([joint])
        torch.cuda.synchronize()
        st = time.time()
        input_device = torch.tensor(input_).to(device)
        output, h_ = model(input_device, h_)
        pre_pos = output.detach().cpu().numpy()[0]
        torch.cuda.synchronize()
        ed = time.time()
        t += (ed - st)
        out.append(pre_pos[0])
    torch.cuda.synchronize()
    end_time = time.time()
    # Calculate total time
    total_time = end_time - start_time
    print("Average model inference time every cycle: ", total_time/len(out))
    print("Model inference time in total: ", total_time)
    print("Model inference time in total: ", t, "Average:", t / len(out))
    out = np.array(out)
    print(out.shape)

    if pos1==0:
        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        xx = np.arange(1, seg + 1)
        integral = [np.sum(cg[i - 1:]) for i in xx]
        plt.plot(xx, cg)
        plt.ylabel("Absolute values of gradients")
        plt.xlabel("Input sequence number")
        plt.tight_layout()
        # plt.savefig(r"./figures/LSTM_input_contribution_seg{}_sr{}.svg".format(seg, sr))
        # plt.show()


    plt.figure(figsize=(18, 12))
    plt.tick_params(labelsize=30)
    if forward:
        plt.plot(data['tendon_disp'][seg:]*6, data['tip_A'][seg:]*90, 'b-', linewidth=2, label="Ground truth")
        plt.plot(data['tendon_disp'][seg:]*6, out[seg:, 0]*90, 'r-', linewidth=2, label="LSTM prediction")
        plt.xlim([-1, 6])
        plt.ylim([0, 90])
        plt.xlabel("Tendon displacement", fontsize=35)
        plt.ylabel("Tip angle azimuth (deg)", fontsize=35)
        plt.title("RMSE:{:.3f} (deg)".format(rmse_norm(out[seg:, 0] * 90, data['tip_A'][seg:] * 90)), fontsize=35)
    else:
        plt.plot(data['tip_A'][seg:] * 90, data['tendon_disp'][seg:] * 6, 'b-', linewidth=2, label="Ground truth")
        plt.plot(data['tip_A'][seg:] * 90, out[seg:, 0] * 6, 'r-', linewidth=2, label="LSTM prediction")
        plt.ylim([-1, 6])
        plt.xlim([0, 90])
        plt.ylabel("Tendon displacement", fontsize=35)
        plt.xlabel("Tip angle azimuth (deg)", fontsize=35)
        plt.title("RMSE:{:.3f} (mm)".format(rmse_norm(out[seg:, 0] * 6, data['tendon_disp'][seg:] * 6)), fontsize=35)
    plt.grid()
    plt.legend(fontsize=30)

    # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.{}Hz_pos{}.jpg".format(test_freq[0], pos1))
    plt.show()

    # plot the training and validation loss


