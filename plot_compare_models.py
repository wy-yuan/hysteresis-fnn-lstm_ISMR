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

def test_LSTM(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1):
    device = "cuda"
    model = LSTMNet(inp_dim=input_dim, num_layers=2)
    sr = sr
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
    rm_init_number = int(rm_init*len(gt))
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse/(max(gt[rm_init_number:])-min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

def test_FNN(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1):
    device = "cuda"
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    sr = sr
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
    print(rm_init_number)
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

if __name__ == '__main__':
    fq, input_dim = False, 1
    # model_foder = "Compare_FNN_NOrsNOfq_LSTM_rsFreq_rmInitalHalfCycle"
    model_foder = "test_buffer_size&sampling_rate_rm_init0_new"
    rm_init = 0
    # model_foder = "test_buffer_size&sampling_rate_rm_init1cycle"
    # rm_init = 1/7

    #  different window size
    LSTM_seg2 = "./checkpoints/Train12345Hz1rep_LSTM_seg2_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch385_best0.0009705254304532141.pt"
    LSTM_seg10 = "./checkpoints/Train12345Hz1rep_LSTM_seg10_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch429_best0.0004312794906606733.pt"
    LSTM_seg20 = "./checkpoints/Train12345Hz1rep_LSTM_seg20_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch494_best0.00024159783231807034.pt"
    LSTM_seg30 = "./checkpoints/Train12345Hz1rep_LSTM_seg30_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch483_best0.000176327922231123.pt"
    LSTM_seg40 = "./checkpoints/Train12345Hz1rep_LSTM_seg40_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch495_best0.00013250087551185921.pt"
    LSTM_seg50 = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch488_best0.00010981084632440833.pt"
    LSTM_seg100 = "./checkpoints/Train12345Hz1rep_LSTM_seg100_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch451_best6.0820018704244984e-05.pt"
    #  different sampling rate
    LSTM_sr10 = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr10_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch449_best8.010477757858071e-05.pt"
    LSTM_sr25 = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch488_best0.00010981084632440833.pt"
    LSTM_sr50 = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr50_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch468_best0.00016143671042987854.pt"
    LSTM_sr100 = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr100_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch377_best0.0002389827078441652.pt"
    #  different window size
    FNN_seg1  = "./checkpoints/Train12345Hz1rep_FNN_seg1_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch50_best0.001193064724235502.pt"
    FNN_seg2  = "./checkpoints/Train12345Hz1rep_FNN_seg2_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch461_best0.0006672149261560176.pt"
    FNN_seg10 = "./checkpoints/Train12345Hz1rep_FNN_seg10_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch439_best0.0002689707871468272.pt"
    FNN_seg20 = "./checkpoints/Train12345Hz1rep_FNN_seg20_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch483_best0.00011461091206294575.pt"
    FNN_seg30 = "./checkpoints/Train12345Hz1rep_FNN_seg30_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch433_best7.225278177007952e-05.pt"
    FNN_seg40 = "./checkpoints/Train12345Hz1rep_FNN_seg40_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch430_best5.554813018486464e-05.pt"
    FNN_seg50 = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch419_best4.5727709091027335e-05.pt"
    FNN_seg100 = "./checkpoints/Train12345Hz1rep_FNN_seg100_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch469_best2.9161683038082238e-05.pt"
    #  different sampling rate
    FNN_sr10 = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr10_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch385_best2.699365524758425e-05.pt"
    FNN_sr25 = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch419_best4.5727709091027335e-05.pt"
    FNN_sr50 = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr50_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch444_best9.54199044659666e-05.pt"
    FNN_sr100 = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr100_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch455_best0.00019113425758016234.pt"


    if not os.path.exists(os.path.join("./figures/", model_foder)):
        os.makedirs(os.path.join("./figures", model_foder))

    test_data = ["0BL_015Hz", "MidBL_015Hz", "EndBL_015Hz", "0BL_045Hz", "MidBL_045Hz", "EndBL_045Hz"]
    # test_data = ["0BL_015Hz"]
    # test_data = ["0BL_02Hz", "MidBL_02Hz", "EndBL_02Hz", "0BL_03Hz", "MidBL_03Hz", "EndBL_03Hz", "0BL_04Hz", "MidBL_04Hz", "EndBL_04Hz"]
    for test_name in test_data:
        data_path = "./tendon_data/45ca_threeTypeDecay/test/{}_repn.txt".format(test_name)
        # data_path = "./tendon_data/45ca_1rep/validate/{}_rep5.txt".format(test_name)
        # data_path = "./tendon_data/45ca_1rep/random_withpause5.txt"

        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax1 = plt.subplots(1, 1, figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.5))

        # data_path = "./tendon_data/45ca_threeTypeDecay/train/EndBL_05hz_rep2.txt"
        model_name = ["FNN_HIB", "LSTM"]
        forward_path = [LSTM_seg2, LSTM_seg10, LSTM_seg20, LSTM_seg30, LSTM_seg40, LSTM_seg50, LSTM_seg100]
        buffer_sizes = [1, 2, 10, 20, 30, 40, 50, 100]
        Forward_rmse_nrmse1 = [100]
        for i in range(len(forward_path)):
            input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(forward_path[i], data_path, seg=buffer_sizes[i], forward=True,
                                                                          fq=fq, input_dim=input_dim)
            Forward_rmse_nrmse1.append(res_rmse)

        plt.tick_params(labelsize=8, pad=0.01, length=2)
        ax1.tick_params(labelsize=8, pad=0.01, length=2)
        ax1.plot(buffer_sizes, Forward_rmse_nrmse1, linewidth=0.6, color='b', marker="o", label='LSTM')
        ax1.set_xlabel('buffer size', fontsize=8)
        ax1.set_ylabel('LSTM RMSE', color='blue', fontsize=8)
        ax1.tick_params('y', colors='blue')
        ax1.set_ylim([0, 3.5])
        ax1.set_xticks(buffer_sizes)

        forward_path = [FNN_seg1, FNN_seg2, FNN_seg10, FNN_seg20, FNN_seg30, FNN_seg40, FNN_seg50, FNN_seg100]
        Forward_rmse_nrmse2 = []
        for i in range(len(forward_path)):
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(forward_path[i], data_path, seg=buffer_sizes[i], forward=True,
                                                                     fq=fq, input_dim=input_dim)
            Forward_rmse_nrmse2.append(res_rmse)

        ax2 = ax1.twinx()
        plt.rcParams['font.family'] = 'Times New Roman'
        ax2.tick_params(labelsize=8, pad=0.01, length=2)
        ax2.plot(buffer_sizes, Forward_rmse_nrmse2, linewidth=0.6, color='r', marker="^", label='FNN-HIB')
        ax2.set_ylabel('FNN-HIB RMSE', color='red', fontsize=8, labelpad=0.01)
        ax2.tick_params('y', colors='red')
        ax2.set_ylim([0, 3.5])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.3, bottom=0.2, left=0.1, right=0.9)
        plt.savefig(r"./figures/{}/Forward_Model_comparison_buffersize_{}.svg".format(model_foder,test_name))
        # plt.show()



# --------------------- --------------------------sampling rate test--------------------------------------------------------#
        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax1 = plt.subplots(1, 1, figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.5))
        # data_path = "./tendon_data/45ca_threeTypeDecay/train/EndBL_05hz_rep2.txt"
        model_name = ["FNN_HIB", "LSTM"]
        forward_path = [LSTM_sr10, LSTM_sr25, LSTM_sr50, LSTM_sr100]
        sample_rate = [10, 25, 50, 100]
        Forward_rmse_nrmse = []
        for i in range(len(forward_path)):
            input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(forward_path[i], data_path, sr=sample_rate[i],
                                                                      forward=True,
                                                                      fq=fq, input_dim=input_dim)
            Forward_rmse_nrmse.append(res_rmse)

        plt.tick_params(labelsize=8, pad=0.01, length=2)
        ax1.tick_params(labelsize=8, pad=0.01, length=2)
        ax1.plot(sample_rate, Forward_rmse_nrmse, linewidth=0.6, color='b', marker="o", label='LSTM')
        ax1.set_xlabel('Sampling rate', fontsize=8)
        ax1.set_ylabel('LSTM RMSE', color='blue', fontsize=8)
        ax1.tick_params('y', colors='blue')
        ax1.set_ylim([0, 5])
        ax1.set_xticks(sample_rate)

        forward_path = [FNN_sr10, FNN_sr25, FNN_sr50, FNN_sr100]
        Forward_rmse_nrmse = []
        for i in range(len(forward_path)):
            input_, gt, output, time, res_rmse, res_nrmse = test_FNN(forward_path[i], data_path, sr=sample_rate[i],
                                                                     forward=True,
                                                                     fq=fq, input_dim=input_dim)
            Forward_rmse_nrmse.append(res_rmse)

        ax2 = ax1.twinx()
        plt.rcParams['font.family'] = 'Times New Roman'
        ax2.tick_params(labelsize=8, pad=0.01, length=2)
        ax2.plot(sample_rate, Forward_rmse_nrmse, linewidth=0.6, color='r', marker="^", label='FNN-HIB')
        ax2.set_ylabel('FNN-HIB RMSE', color='red', fontsize=8, labelpad=0.01)
        ax2.tick_params('y', colors='red')
        ax2.set_ylim([0, 5])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.3, bottom=0.2, left=0.1, right=0.9)
        plt.savefig(r"./figures/{}/Forward_Model_comparison_samplingRate_{}.svg".format(model_foder, test_name))
        # plt.show()
