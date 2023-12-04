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

def test_FNN(pt_path, data_path, seg=50, forward=True, fq=False, input_dim=1):
    device = "cuda"
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    sr = 25
    seg = seg
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
    for i in range(data['tendon_disp'].shape[0] - seg):
        joint = joints[i + 1:i + seg + 1, 0:input_dim]
        pre_pos = pos[i:i + seg, 0:1]
        if pos1 == 1:
            input_ = np.hstack([joint, pre_pos])
        else:
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
    res_rmse = rmse_norm(output, gt)
    res_nrmse = res_rmse / (max(gt) - min(gt))
    return input, gt, output, data['time'], res_rmse, res_nrmse

if __name__ == '__main__':
    device = "cuda"
    path = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3/TC_FNN_L2_bs16_epoch77_best0.00018853881680396808.pt"
    path = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr50_loss1_epoch1000/TC_FNN_L2_bs16_epoch405_best0.00011245763385482456.pt"

    FNN_forward = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg1_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch322_best0.0012303077845923825.pt"
    # FNN_inverse = "./checkpoints/Inverse_TC_FNN_45ca_threeTypeDecay_seg1_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch385_best0.002596977944650679.pt"
    # FNN50_forward = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch939_best8.775984691583514e-05.pt"
    # FNN50_inverse = "./checkpoints/Inverse_TC_FNN_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch228_best0.0002140288114986203.pt"

    # fq, input_dim = False, 1
    # path = "./checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch490_best0.6320451708976179.pt"
    path = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch278_best1.8999528847634792.pt"

    fq, input_dim = True, 2
    path = "./checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_FNN_L2_bs16_epoch358_best0.8296004019288066.pt"
    # path = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_FNN_L2_bs16_epoch144_best2.4323931725975734.pt"
    # path = FNN_forward
    forward = True
    pos1 = 0
    sr = 100
    seg = 50
    # fq, input_dim = True, 2
    # fq, input_dim = False, 1
    # load trained model
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    model.load_state_dict(torch.load(path, map_location=device))
    model.cuda()
    model.eval()


    path = "./tendon_data/45ca_threeTypeDecay/validate/0BL_02hz_rep5.txt"
    path = "./tendon_data/45ca_threeTypeDecay/validate/0BL_045Hz_repn.txt"
    # path = "./tendon_data/45ca/train/0_1hz_rep4.txt"
    data_path = os.path.join(path)
    data = load_data(data_path, seg, sample_rate=sr)

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
    if fq:
        freq = data['freq'][:, np.newaxis].astype("float32")
        joints = np.concatenate([joints, freq], axis=1)
    pos = data['tip_A'][:, np.newaxis].astype("float32")

    out = []
    for i in range(data['tendon_disp'].shape[0]-seg):
        joint = joints[i+1:i + seg+1, 0:input_dim]
        pre_pos = pos[i:i+seg, 0:1]
        if pos1 == 1:
            input_ = np.hstack([joint, pre_pos])
        else:
            input_ = joint
        output = model(torch.tensor([input_]).to(device))
        predict_pos = output.detach().cpu().numpy()[0]
        out.append(predict_pos[0])
    out = np.array(out)
    print(out.shape)

    plt.figure(figsize=(18, 12))
    plt.tick_params(labelsize=30)
    if forward:
        plt.plot(data['tendon_disp'][seg:] * 6, data['tip_A'][seg:] * 90, 'b-', linewidth=4, label="Ground truth")
        plt.plot(data['tendon_disp'][seg:] * 6, out[:, 0] * 90, 'r-', linewidth=4, label="FNN prediction")
        plt.xlim([-1, 6])
        plt.ylim([0, 90])
        plt.xlabel("Tendon displacement", fontsize=35)
        plt.ylabel("Tip angle azimuth (deg)", fontsize=35)
        plt.title("RMSE:{:.3f} (deg)".format(rmse_norm(out[:, 0] * 90, data['tip_A'][seg:] * 90)), fontsize=35)
    else:
        plt.plot(data['tip_A'][seg:] * 90, data['tendon_disp'][seg:] * 6, 'b-', linewidth=4, label="Ground truth")
        plt.plot(data['tip_A'][seg:] * 90, out[:, 0] * 6, 'r-', linewidth=4, label="FNN prediction")
        plt.ylim([-1, 6])
        plt.xlim([0, 90])
        plt.ylabel("Tendon displacement", fontsize=35)
        plt.xlabel("Tip angle azimuth (deg)", fontsize=35)
        plt.title("RMSE:{:.3f} (mm)".format(rmse_norm(out[:, 0] * 6, data['tendon_disp'][seg:] * 6)), fontsize=35)
    plt.grid()
    plt.legend(fontsize=30)
    # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.{}Hz_pos{}.jpg".format(test_freq[0], pos1))
    plt.show()

    # plot the training and validation loss
    acc_file = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3/TC_FNN_L2_acc_bs16_epoch100.pkl"
    # acc_file = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg1_sr100_rsFalse_lr1e-3/TC_FNN_L2_acc_bs16_epoch100.pkl"
    acc_file = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3_loss1/TC_LSTM_L2_acc_bs16_epoch100.pkl"
    acc_file = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3_loss10000/TC_LSTM_L2_acc_bs16_epoch100.pkl"
    acc = pickle.load(open(acc_file, "rb"))
    plt.figure(figsize=(20, 10))
    plt.plot(acc['train'], '-', label='train')
    plt.plot(acc['test'], '-', label='test')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('./results/acc_{}.jpg')
    # plt.show()
