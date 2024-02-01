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
from plot_compare_models import test_LSTM
from plot_compare_models import test_FNN


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
    interp_time = np.arange(time[0], time[-1], 1 / sample_rate)
    tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
    tip_A_resample = np.interp(interp_time, time, tip_A)

    # normalization to [0, 1] and pad -1
    tendon_disp = np.hstack([-np.ones(seg), tendon_disp_resample / 6])
    tip_A = np.hstack([-np.ones(seg), tip_A_resample / 90])
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'freq': freq}


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

    model_name, model_path, forward = "forward_LSTM_40", forward_LSTM_path, True
    # model_name, model_path, forward = "forward_FNN_40", forward_FNN_path, True
    # model_name, model_path, forward = "inverse_LSTM_50", inverse_LSTM_path, False
    # model_name, model_path, forward = "inverse_FNN", inverse_FNN_path, False
    pos1 = 0
    sr = 25
    seg = 50
    fq, input_dim = False, 1
    act = None
    rm_init = 0
    # path = "./tendon_data/innerTSlack40/train/Random_win30_3.txt"
    path = "./tendon_data/innerTSlack40/validate/0BL_015Hz_1.txt"
    path = "./tendon_data/innerTSlack50/test/Random_win30_21.txt"  # 29, 30? 21, 23, 25
    # path = "./tendon_data/innerTSlack60/train/Random_win30_40.txt"  # 29, 30? 21, 23, 25
    path = "./tendon_data/innerTSlack50/test/ablation_motion_2.txt"  # 29, 30? 21, 23, 25
    # path = r"D:\yuan\Dropbox (BCH)\bch\projects\hysteresis-fnn-lstm_ISMR\tendon_data\Train12345Hz_1rep_fakePause\train\random_pause1_35.txt"
    motion_name = os.path.basename(path).split(".")[0]
    data_path = path
    if "LSTM" in model_name:
        input_, gt, output, time, res_rmse, res_nrmse = test_LSTM(model_path, data_path, seg=seg, forward=forward,
                                                                  fq=fq, input_dim=input_dim, rm_init=rm_init, numl=1, h=64)
    else:
        input_, gt, output, time, res_rmse, res_nrmse = test_FNN(model_path, data_path, seg=seg, forward=forward, fq=fq,
                                                                 input_dim=input_dim, rm_init=rm_init)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 * 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(input_, gt, linewidth=0.8, label="Ground truth")
    plt.plot(input_, output, linewidth=0.8, label=model_name + "_model")
    # plt.ylim([-0.5, 6.5])
    # plt.xlim([0, 90])
    if forward:
        plt.xlabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
        plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
        plt.title("RMSE:{:.3f} (deg)".format(rmse_norm(output, gt)), fontsize=8)
    else:
        plt.ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
        plt.xlabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
        plt.title("RMSE:{:.3f} (mm)".format(rmse_norm(output, gt)), fontsize=8)
    plt.legend(fontsize=8, frameon=False)

    ax1 = plt.subplot(2, 1, 2)
    ax1.tick_params(labelsize=8, pad=0.01, length=2)
    ax1.plot(time, gt, linewidth=0.8, label="Ground truth")
    ax1.plot(time, output, linewidth=0.8, label=model_name + "_model")
    ax1.set_xlabel("Time (s)", fontsize=8, labelpad=0.01)
    if forward:
        ax1.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
        ax1.set_ylim([0, 90])
    else:
        ax1.set_ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
        ax1.set_ylim([-0.1, 6.1])
    plt.legend(fontsize=8, frameon=False)

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=8, pad=0.01, length=2)
    ax2.plot(time, input_, linewidth=0.8, color='g')
    ax2.tick_params('y', colors='g')

    if forward:
        ax2.set_ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01, color='g')
        ax2.set_ylim([-0.1, 6.1])
    else:
        ax2.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01, color='g')
        ax2.set_ylim([0, 90])
    plt.tight_layout()
    # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.{}Hz_pos{}.jpg".format(test_freq[0], pos1))
    plt.savefig("./figures/innerTendonSlack/{}_{}.png".format(model_name, motion_name), dpi=600)
    plt.show()

# for i in range(data['tendon_disp'].shape[0]):
#     joint = angles[i:i + 1, 0:input_dim]
#     input_device = torch.tensor(np.array([joint])).to(device)
#     output, h_ = inv_model(input_device, h_)
#     pre_tendon = output.detach().cpu().numpy()[0]
#     out.append(pre_tendon[0])

# # predict tendon disp. with designated tip angle
# if i > 640:
#     feedback_out, _ = inv_model(input_device, feedback_h_)
# else:
#     feedback_out, _ = inv_model(input_device, _)
# pre_tendon_feedback = feedback_out.detach().cpu().numpy()[0]
# out_feedback.append(pre_tendon_feedback[0])
# # use forward kinematics model to simulate tip angles
# tendon = pre_tendon_feedback
# tendon_device = torch.tensor(np.array([tendon])).to(device)
# forward_out, forward_h_ = model(tendon_device, forward_h_)
# pre_angle = forward_out
# pre_angles.append(pre_angle.detach().cpu().numpy()[0][0])
#
# o_, feedback_h_ = inv_model(pre_angle, feedback_h_)
