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
    FNN_forward = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg1_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch322_best0.0012303077845923825.pt"
    FNN_inverse = "./checkpoints/Inverse_TC_FNN_45ca_threeTypeDecay_seg1_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch385_best0.002596977944650679.pt"
    # FNN50_forward = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch490_best8.868874949721692e-05.pt"
    # FNN50_inverse = "./checkpoints/Inverse_TC_FNN_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch228_best0.0002140288114986203.pt"
    # LSTM_forward = "./checkpoints/TC_LSTM_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_LSTM_L2_bs16_epoch351_best0.00026320803928698715.pt"
    # LSTM_inverse = "./checkpoints/Inverse_TC_LSTM_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_LSTM_L2_bs16_epoch326_best0.0004697214213076312.pt"

    fq, input_dim = False, 1
    # model_foder = "Compare_FNN_NOrsNOfq_LSTM_rsFreq_rmInitalHalfCycle"
    model_foder = "sr25_NOrsNOfq_loss10000_figure_for_paper"
    FNN50_forward = "./checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch490_best0.6320451708976179.pt"
    FNN50_inverse = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch278_best1.8999528847634792.pt"
    LSTM_forward = "./checkpoints/FixValidateData_TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch248_best1.3202891629189253.pt"
    LSTM_inverse = "./checkpoints/FixValidateData_Inverse_TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch352_best2.2775139451026916.pt"

    # FNN50_forward = "./checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch500.pt"
    # FNN50_inverse = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch300.pt"
    # LSTM_forward = "./checkpoints/FixValidateData_TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch500.pt"
    # LSTM_inverse = "./checkpoints/FixValidateData_Inverse_TC_LSTM_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch400.pt"

    # fq, input_dim = True, 2
    # model_foder = "sr100_rsFreq_loss10000_rmInitalHalfCycle"
    # FNN50_forward = "./checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_FNN_L2_bs16_epoch358_best0.8296004019288066.pt"
    # FNN50_inverse = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_FNN_L2_bs16_epoch279_best2.390517575873269.pt"
    # LSTM_forward = "./checkpoints/FixValidateData_TC_LSTM_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch425_best1.3930550070564731.pt"
    # LSTM_inverse = "./checkpoints/FixValidateData_Inverse_TC_LSTM_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_LSTM_L2_bs16_epoch363_best2.370356702337078.pt"

    rm_init = 0
    if not os.path.exists(os.path.join("./figures/", model_foder)):
        os.makedirs(os.path.join("./figures", model_foder))
    test_data = ["MidBL_045Hz"]
    # test_data = ["0BL_02Hz", "MidBL_02Hz", "EndBL_02Hz", "0BL_03Hz", "MidBL_03Hz", "EndBL_03Hz", "0BL_04Hz", "MidBL_04Hz", "EndBL_04Hz"]
    for test_name in test_data:
        # test_name = "MidBL_015Hz"
        data_path = "./tendon_data/45ca_threeTypeDecay/validate/{}_repn.txt".format(test_name)
        # data_path = "./tendon_data/45ca_1rep/validate/{}_rep5.txt".format(test_name)
        # data_path = "./tendon_data/45ca_1rep/random_withpause5.txt"

        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        plt.rcParams['font.family'] = 'Times New Roman'

        Forward_rmse_nrmse = []
        Inverse_rmse_nrmse = []
        # data_path = "./tendon_data/45ca_threeTypeDecay/train/EndBL_05hz_rep2.txt"
        model_name = ["FNN", "FNN_HIB", "LSTM"]
        forward_path = [FNN_forward, FNN50_forward, LSTM_forward]
        inverse_path = [FNN_inverse, FNN50_inverse, LSTM_inverse]
        color_list = ['r', 'g', 'darkorange']
        for i in range(3):
            print(i)
            if i == 0:
                input_, gt, output, time, res_rmse, res_nrmse = test_FNN(forward_path[i], data_path, seg=1, forward=True,
                                                                         fq=False, input_dim=1)
            elif i == 1:
                # fq, input_dim = False, 1
                input_, gt, output, time, res_rmse, res_nrmse = test_FNN(forward_path[i], data_path, seg=50, forward=True,
                                                                         fq=fq, input_dim=input_dim)
            elif i == 2:
                # fq, input_dim = True, 2
                input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(forward_path[i], data_path, forward=True,
                                                                          fq=fq, input_dim=input_dim)

            Forward_rmse_nrmse.append(res_rmse)
            Forward_rmse_nrmse.append(res_nrmse)

            plt.subplot(2, 3, i+1)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            plt.plot(input_, gt, linewidth=0.6, color='b', label='GT')
            plt.plot(input_, output, linewidth=0.6, color=color_list[i], label=model_name[i])
            plt.xlabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
            plt.ylabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
            # plt.title("RMSE:{:.3f} (deg)".format(res_rmse), fontsize=8)
            plt.ylim([10, 90])
            plt.xlim([-0.5, 6.5])
            plt.xticks([0, 3, 6])
            plt.yticks([10, 30, 50, 70, 90])
            plt.legend(fontsize=6, frameon=False, loc='lower right', handlelength=1.2) # ,loc='upper left',

            plt.subplot(2, 1, 2)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            if i == 0:
                plt.plot(time, gt, linewidth=0.6, color='b', label='GT')
            plt.plot(time, output, linewidth=0.6, color=color_list[i], label=model_name[i])
            plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
            plt.ylabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
            plt.yticks([10, 30, 50, 70, 90])
            plt.legend(fontsize=6, frameon=False, loc='upper right', ncol=2)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.3, bottom=0.1, left=0.08, right=0.95)
        plt.savefig(r"./figures/{}/Forward_Model_comparison_{}.svg".format(model_foder,test_name))
        plt.show()

        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.3))
        plt.rcParams['font.family'] = 'Times New Roman'

        inverse_path = [FNN_inverse, FNN50_inverse, LSTM_inverse]
        color_list = ['r', 'g', 'darkorange']
        for i in range(3):
            print(i)
            if i == 0:
                input_, gt, output, time, res_rmse, res_nrmse = test_FNN(inverse_path[i], data_path, seg=1, forward=False,
                                                                         fq=False, input_dim=1)
            elif i == 1:
                # fq, input_dim = False, 1
                input_, gt, output, time, res_rmse, res_nrmse = test_FNN(inverse_path[i], data_path, seg=50, forward=False,
                                                                         fq=fq, input_dim=input_dim)
            elif i == 2:
                # fq, input_dim = True, 2
                input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(inverse_path[i], data_path, forward=False,
                                                                          fq=fq, input_dim=input_dim)

            Inverse_rmse_nrmse.append(res_rmse)
            Inverse_rmse_nrmse.append(res_nrmse)

            plt.subplot(2, 3, i+1)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            plt.plot(input_, gt, linewidth=0.8, color='b', label='GT')
            plt.plot(input_, output, linewidth=0.8, color=color_list[i], label=model_name[i])
            plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
            plt.xlabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
            # plt.title("RMSE:{:.3f} (mm)".format(res_rmse), fontsize=8)
            plt.xlim([10, 90])
            plt.ylim([-0.5, 6.5])
            plt.yticks([0, 3, 6])
            plt.xticks([10, 30, 50, 70, 90])
            plt.legend(fontsize=6, frameon=False, loc='lower right', handlelength=1.2) # ,loc='upper left',

            plt.subplot(2, 1, 2)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            if i == 0:
                plt.plot(time, gt, linewidth=0.8, color='b', label='GT')
            plt.plot(time, output, linewidth=0.8, color=color_list[i], label=model_name[i])
            plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
            plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
            # plt.yticks([10, 30, 50, 70, 90])
            plt.legend(fontsize=6, frameon=False, loc='upper right', ncol=2)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.3, bottom=0.1, left=0.08, right=0.95)
        plt.savefig(r"./figures/{}/Inverse_Model_comparison_{}.svg".format(model_foder, test_name))
        # plt.show()

        header_name = "FNN_rmse     FNN_nrmse      FNN_HIB_rmse     FNN_HIB_nrmse      LSTM_rmse     LSTM_nrmse"
        forward = np.array(Forward_rmse_nrmse)
        Inverse = np.array(Inverse_rmse_nrmse)
        rmse_nrmse = np.vstack([forward, Inverse])
        np.savetxt("./figures/{}/rmse_nrmse_{}.txt".format(model_foder, test_name), rmse_nrmse, fmt='%.3f', header=header_name, delimiter="\t\t")

