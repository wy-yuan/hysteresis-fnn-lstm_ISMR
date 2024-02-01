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

def test_LSTM(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, numl=2):
    device = "cuda"
    model = LSTMNet(inp_dim=input_dim, num_layers=numl)
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

def test_FNN(pt_path, data_path, seg=50, sr=25, forward=True, fq=False, input_dim=1, numl=2):
    device = "cuda"
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg, num_layers=numl)
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
    # print(rm_init_number)
    res_rmse = rmse_norm(output[rm_init_number:], gt[rm_init_number:])
    res_nrmse = res_rmse / (max(gt[rm_init_number:]) - min(gt[rm_init_number:]))
    return input, gt, output, data['time'], res_rmse, res_nrmse

if __name__ == '__main__':
    fq, input_dim = False, 1
    # model_foder = "SinusforTraining_model_layer1_comparison"
    # layers = 1
    # FNN_forward_path = "./checkpoints/innerTSlackSinus_FNN_layer1_seg1_sr25/TC_FNN_L1_bs16_epoch366_best0.0009500099917266013.pt"
    # FNN50_forward_path = "./checkpoints/innerTSlackSinus_FNN_layer1_seg50_sr25/TC_FNN_L1_bs16_epoch460_best9.662615048000589e-05.pt"
    # LSTM_forward_path = "./checkpoints/innerTSlackSinus_LSTM_layer1_seg50_sr25/TC_LSTM_L1_bs16_epoch321_best0.0001722811882375806.pt"
    # LSTM_forward_40_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq/TC_LSTM_L1_bs16_epoch184_best0.00021742242097388953.pt"  #may retrain
    # LSTM_forward_40_loss_path = "./checkpoints/innerTSlack40_LSTM_layer1_seg50_sr25_NOrsNOfq_stdRatio0.0001/TC_LSTM_L1_bs16_epoch350.pt"

    model_foder = "Model_layer2_comparison_step_motion"
    layers = 2
    FNN_forward_path = "./checkpoints/innerTSlackSinus_FNN_layer2_seg1_sr25/TC_FNN_L2_bs16_epoch366_best0.0009479064877160778.pt"
    FNN50_forward_path = "./checkpoints/innerTSlack_layer2/innerTSlackSinus_FNN_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch165_best8.132606441023878e-05.pt"
    LSTM_forward_path = "./checkpoints/innerTSlack_layer2/innerTSlackSinus_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch418_best0.00015659084497851187.pt"
    LSTM_forward_40_path = "./checkpoints/innerTSlack_layer2/innerTSlack40_LSTM_seg50_sr25_NOrsNOfq/TC_LSTM_L2_bs16_epoch489_best0.00018550869560660056.pt"
    LSTM_forward_40_loss_path = "./checkpoints/innerTSlack40_LSTM_layer2_seg50_sr25_stdRatio1e-4_delta1e-4/TC_LSTM_L2_bs16_epoch229_best0.0003166038428418457.pt"

    rm_init = 0
    if not os.path.exists(os.path.join("./figures/", model_foder)):
        os.makedirs(os.path.join("./figures", model_foder))
    # test_data = ["MidBL_045Hz"]
    # test_data = ["0BL_015Hz", "MidBL_015Hz", "EndBL_015Hz", "0BL_045Hz", "MidBL_045Hz", "EndBL_045Hz"]
    # test_data = ["Random_win30_21", "Random_win30_22", "Random_win30_23", "Random_win30_24", "Random_win30_25", "ablation_motion_2"]
    test_data = ["step_motion_1", "step_motion_2", "step_motion_3", "step_motion_4", "step_motion_5"]
    test_data = ["step_motion2_1", "step_motion2_2", "step_motion2_3", "step_motion2_4", "step_motion2_5",
                 "step_motion2_6", "step_motion2_7", "step_motion2_8", "step_motion2_9", "step_motion2_10"]

    for test_name in test_data:
        # data_path = "./tendon_data/innerTSlackSinus/validate/{}_1.txt".format(test_name)
        data_path = "./tendon_data/innerTSlack40/test/{}.txt".format(test_name)

        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        plt.rcParams['font.family'] = 'Times New Roman'

        Forward_rmse_nrmse = []
        model_name = ["FNN", "FNN_HIB", "LSTM"]
        forward_path = [FNN_forward_path, FNN50_forward_path, LSTM_forward_path]
        color_list = ['r', 'g', 'darkorange']
        for i in range(3):
            print(i)
            if i == 0:
                input_, gt, output, time, res_rmse, res_nrmse = test_FNN(forward_path[i], data_path, seg=1, forward=True,
                                                                         fq=False, input_dim=1, numl=layers)
            elif i == 1:
                # fq, input_dim = False, 1
                input_, gt, output, time, res_rmse, res_nrmse = test_FNN(forward_path[i], data_path, seg=50, forward=True,
                                                                         fq=fq, input_dim=input_dim, numl=layers)
            elif i == 2:
                # fq, input_dim = True, 2
                input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(forward_path[i], data_path, forward=True,
                                                                          fq=fq, input_dim=input_dim, numl=layers)

            Forward_rmse_nrmse.append(res_rmse)
            Forward_rmse_nrmse.append(res_nrmse)

            plt.subplot(2, 1, 1)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            if i == 0:
                plt.plot(time, gt, linewidth=0.6, color='b', label='GT')
            plt.plot(time, output, linewidth=0.6, color=color_list[i], label=model_name[i])
            plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
            plt.ylabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
            plt.yticks([50, 60, 70])
            plt.ylim([43, 70])
            # plt.xlim([15, 32])
            plt.xlim([5, 11])
            plt.legend(fontsize=6, frameon=False, loc='upper right', ncol=2)

            plt.subplot(2, 3, i+1+3)
            plt.tick_params(labelsize=8, pad=0.01, length=2)
            plt.plot(input_, gt, linewidth=0.6, color='b', label='GT')
            plt.plot(input_, output, linewidth=0.6, color=color_list[i], label=model_name[i])
            plt.xlabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
            plt.ylabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
            # plt.title("RMSE:{:.3f} (deg)".format(res_rmse), fontsize=8)
            plt.ylim([15, 85])
            plt.xlim([-0.1, 6.1])
            plt.xticks([0, 3, 6])
            # plt.yticks([10, 30, 50, 70, 90])
            plt.yticks([15, 50, 85])
            plt.legend(fontsize=6, frameon=False, loc='lower right', handlelength=1.2) # ,loc='upper left',


        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.3, bottom=0.1, left=0.08, right=0.95)
        plt.savefig(r"./figures/{}/Forward_Model_comparison_{}.svg".format(model_foder,test_name))
        # plt.show()

        LSTM_rmse_nrmse = []
        model_name = ["S", "SS", "RL"]
        forward_path = [LSTM_forward_path, LSTM_forward_40_path, LSTM_forward_40_loss_path]
        color_list = ['darkorange', 'blueViolet', "limegreen"]  #DeepSkyBlue, DodgerBlue, brown

        # plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        # plt.rcParams['font.family'] = 'Times New Roman'
        gs_kw = dict(width_ratios=[1, 1], height_ratios=[2, 1])
        # fig, axd = plt.subplot_mosaic([['up1','up2','up3'],['down','down','down']], gridspec_kw=gs_kw,
        #                               figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4), layout="constrained") #layout="constrained"
        fig, axd = plt.subplot_mosaic([['down', 'down'], ['up1', 'up2']], gridspec_kw=gs_kw,
                                      figsize=(88.9 / 25.4, 88.9 / 25.4 * 1.2), layout="constrained")  # layout="constrained"
        plt.rcParams['font.family'] = 'Times New Roman'
        for k, ax in axd.items():
            print(k, ax)
        index = 0
        for i in range(3):
            print(i)
            input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(forward_path[i], data_path, forward=True,
                                                                          fq=fq, input_dim=input_dim, numl=layers)

            LSTM_rmse_nrmse.append(res_rmse)
            LSTM_rmse_nrmse.append(res_nrmse)

            if i != 1:
                # plt.subplot(2, 3, i)
                plt.subplot(2, 2, index+1+2)
                index += 1
                plt.tick_params(labelsize=8, pad=0.01, length=2)
                plt.plot(input_, gt, linewidth=0.8, color='b', label='GT')
                plt.plot(input_, output, linewidth=0.8, color=color_list[i], label=model_name[i])
                plt.xlabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
                plt.ylabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
                # plt.title("RMSE:{:.3f} (deg)".format(res_rmse), fontsize=8)
                plt.ylim([15, 90])
                plt.xlim([-0.2, 6.2])
                plt.xticks([0, 3, 6])
                # plt.yticks([10, 30, 50, 70, 90])
                plt.yticks([15, 50, 85])
                plt.legend(fontsize=7, frameon=False, loc='lower right', handlelength=1.2)  # ,loc='upper left',

            # plt.subplot(2, 1, 2)
            axd["down"].tick_params(labelsize=8, pad=0.01, length=2)
            if i == 0:
                axd["down"].plot(time, gt, linewidth=0.8, color='b', label='GT')
            axd["down"].plot(time, output, linewidth=0.8, color=color_list[i], label=model_name[i])
            axd["down"].plot(time, abs(output-gt), "--", linewidth=0.8, color=color_list[i], label=model_name[i]+" error")
            axd["down"].set_xlabel('Time (s)', fontsize=8, labelpad=0.01)
            axd["down"].set_ylabel('Tip angle (deg)', fontsize=8, labelpad=0.01)
            axd["down"].set_yticks([0, 20, 40, 60, 80, 100])
            axd["down"].set_ylim([-1, 100])
            axd["down"].legend(fontsize=7, frameon=False, loc='upper right', ncol=4)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.3, bottom=0.1, left=0.08, right=0.95)
        plt.savefig(r"./figures/{}/Forward_LSTM_comparison_{}.svg".format(model_foder, test_name))

        header_name = "FNN_rmse     FNN_nrmse      FNN_HIB_rmse     FNN_HIB_nrmse      LSTM_rmse     LSTM_nrmse"
        forward = np.array(Forward_rmse_nrmse)
        Inverse = np.array(LSTM_rmse_nrmse)
        rmse_nrmse = np.vstack([forward, Inverse])
        np.savetxt("./figures/{}/rmse_nrmse_{}.txt".format(model_foder, test_name), rmse_nrmse, fmt='%.3f', header=header_name, delimiter="\t\t")

