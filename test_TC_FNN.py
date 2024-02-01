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
from queue import Queue
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
        pre_pos = data['tip_A'][:, np.newaxis].astype("float32")[0:0 + seg, 0:1]
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
        pre_pos = data['tendon_disp'][:, np.newaxis].astype("float32")[0:0 + seg, 0:1]
    if fq:
        freq = data['freq'][:, np.newaxis].astype("float32")
        joints = np.concatenate([joints, freq], axis=1)

    out = []
    for i in range(data['tendon_disp'].shape[0] - seg):
        joint = joints[i + 1:i + seg + 1, 0:input_dim]
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

def check_weights(net):
    fc1 = 0
    for name, param in net.named_parameters():
        if 'weight' in name:
            print(f'Layer: {name}, Size: {param.size()}')
            print(param)
            print('\n' + '-' * 50 + '\n')
            if 'fc1' in name:
                fc1 = param.detach().cpu().numpy()
    print(fc1.shape)
    x = np.sum(fc1, axis=0)
    print(x.shape, x)
    plt.plot(abs(x), label="|weights|")
    plt.plot(x, label="weights")
    plt.legend()
    plt.show()

def compute_gradient(model, input_array, seg=50):
    # input_tensor = torch.randn(1, seg, requires_grad=True)
    # input_tensor = torch.ones(1, seg, requires_grad=True)
    input_tensor = torch.tensor(input_array, requires_grad=True)
    output = model(input_tensor)
    # Calculate the contribution of each input (gradient of the loss with respect to the input)
    contributions = grad(output, input_tensor)[0]
    x = contributions.detach().cpu().numpy()[0]
    # print(x.shape)
    return x
    # plt.plot(abs(x), label="|x|")
    # plt.plot(x, label="x")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    path = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3/TC_FNN_L2_bs16_epoch77_best0.00018853881680396808.pt"
    path = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr50_loss1_epoch1000/TC_FNN_L2_bs16_epoch405_best0.00011245763385482456.pt"

    FNN_forward = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg1_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch322_best0.0012303077845923825.pt"
    # FNN_inverse = "./checkpoints/Inverse_TC_FNN_45ca_threeTypeDecay_seg1_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch385_best0.002596977944650679.pt"
    # FNN50_forward = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch939_best8.775984691583514e-05.pt"
    # FNN50_inverse = "./checkpoints/Inverse_TC_FNN_45ca_threeTypeDecay_seg50_sr25_loss1_epoch1000/TC_FNN_L2_bs16_epoch228_best0.0002140288114986203.pt"

    # fq, input_dim = False, 1
    path = "../hysteresis_ISMR/checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch490_best0.6320451708976179.pt"
    # path = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr25_NOrsNOfq_loss10000_epoch500/TC_FNN_L2_bs16_epoch278_best1.8999528847634792.pt"

    # fq, input_dim = True, 2
    # path = "./checkpoints/FixValidateData_TC_FNN_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_FNN_L2_bs16_epoch358_best0.8296004019288066.pt"
    # path = "./checkpoints/FixValidateData_Inverse_TC_FNN_45ca_1rep_seg50_sr100_rsFreq_loss10000_epoch500/TC_FNN_L2_bs16_epoch144_best2.4323931725975734.pt"
    # path = FNN_forward

    path = "./checkpoints/FNN_45ca1rep_seg50_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch456_best0.6284833196550608.pt"
    path = "./checkpoints/FNN_45ca1rep_seg2_sr25_NOrsNOfq/TC_FNN_L2_bs16_epoch327_best6.251717589548882.pt"

    # path = "./checkpoints/FNN_Train135Hz3rep_seg50_sr25_NOrsNOfq_TCdataset/TC_FNN_L2_bs16_epoch360_best2.721998561075078e-05.pt"
    # path = "./checkpoints/FNN_Train135Hz3rep_seg50_sr25_NOrsNOfq_TCdataset_wde-5/TC_FNN_L2_bs16_epoch363_best8.869190335466318e-05.pt"
    # path = "./checkpoints/FNN_Train135Hz3rep_seg50_sr25_NOrsNOfq_TCdataset_wde-7/TC_FNN_L2_bs16_epoch384_best2.5948377318956908e-05.pt"
    # path = "./checkpoints/FNN_45ca1rep_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch346_best7.689633991958544e-05.pt"

    #  different window size
    path = "./checkpoints/Train12345Hz1rep_FNN_seg1_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch50_best0.001193064724235502.pt"
    path = "./checkpoints/Train12345Hz1rep_FNN_seg2_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch461_best0.0006672149261560176.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg10_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch439_best0.0002689707871468272.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg20_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch483_best0.00011461091206294575.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg30_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch433_best7.225278177007952e-05.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg40_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch430_best5.554813018486464e-05.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch419_best4.5727709091027335e-05.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg100_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch469_best2.9161683038082238e-05.pt"

    #  different sampling rate
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr10_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch385_best2.699365524758425e-05.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch419_best4.5727709091027335e-05.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr50_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch444_best9.54199044659666e-05.pt"
    # path = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr100_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch455_best0.00019113425758016234.pt"

    path = "./checkpoints/Train12345Hz1rep_INV_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch486_best0.00015277773349367494.pt"
    # path = "./checkpoints/Train12345Hz1rep_INV_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd_pos1/TC_FNN_L2_bs16_epoch449_best0.0001882735092507361.pt"

    path = "./checkpoints/longPause_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch62_best0.00035399518612578857.pt"
    path = "./checkpoints/longPause_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch60_best0.0002575914480122126.pt"
    path = "./checkpoints/longPause_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch44_best0.0005857760914881572.pt"
    path = "./checkpoints/longPause_INV_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch47_best0.0005895499229062175.pt"

    forward = False
    pos1 = 0
    sr = 25
    seg = 50
    # fq, input_dim = True, 2
    fq, input_dim = False, 1
    # load trained model
    device = "cpu"
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    model.load_state_dict(torch.load(path, map_location=device))
    # model.cuda()
    model.eval()

    path = "./tendon_data/45ca_1rep/validate/0BL_01hz_rep5.txt"
    path = "./tendon_data/45ca_threeTypeDecay/test/EndBL_045Hz_repn.txt"
    path = "./tendon_data/45ca_1rep/random_withpause5.txt"
    # path = "./tendon_data/45ca/train/0_1hz_rep4.txt"
    # path = "./tendon_data/Train135Hz_1rep/validate/EndBL_01hz_rep5.txt"
    # path = "./tendon_data/Train135Hz_1rep/validate/0BL_03hz_rep4.txt"
    # path = "./tendon_data/Train135Hz_1rep/validate/0BL_05Hz_rep4.txt"
    # path = "./tendon_data/Train135Hz_1rep/test/0BL_02Hz_rep2.txt"
    # path = "./tendon_data/Train135Hz_1rep/test/0BL_04hz_rep2.txt"
    data_path = os.path.join(path)
    data = load_data(data_path, seg, sample_rate=sr)

    if forward:
        joints = data['tendon_disp'][:, np.newaxis].astype("float32")
        pos = data['tip_A'][:, np.newaxis].astype("float32")
        pre_pos = pos[0:0 + seg, 0:1]
    else:
        joints = data['tip_A'][:, np.newaxis].astype("float32")
        pos = data['tendon_disp'][:, np.newaxis].astype("float32")
        pre_pos = pos[0:0 + seg, 0:1]
    if fq:
        freq = data['freq'][:, np.newaxis].astype("float32")
        joints = np.concatenate([joints, freq], axis=1)

    # check the contribution of each input to the output
    # check_weights(model)

    out = []
    torch.cuda.synchronize()
    start_time = time.time()
    t = 0
    cg = 0
    for i in range(data['tendon_disp'].shape[0]-seg):
        if i > 0 and pos1!=1:
            contribute = (compute_gradient(model, np.array([joints[i + 1:i + seg + 1, 0:input_dim]]), seg=seg))
            cg += np.abs(contribute)
        joint = joints[i+1:i + seg+1, 0:input_dim]
        # pre_pos = pos[i:i + seg, 0:1]
        if pos1 == 1:
            input_ = np.hstack([joint, pre_pos])[np.newaxis, :]
        else:
            input_ = np.array([joint])
        torch.cuda.synchronize()
        st = time.time()
        input_device = torch.tensor(input_).to(device)
        output = model(input_device)
        torch.cuda.synchronize()
        predict_pos = output.detach().cpu().numpy()[0]
        ed = time.time()
        t += (ed - st)
        out.append(predict_pos[0])
        pre_pos = np.vstack([pre_pos[1:], predict_pos])

    torch.cuda.synchronize()
    end_time = time.time()
    # Calculate total time
    total_time = end_time - start_time
    print("Average model inference time every cycle: ", total_time / len(out))
    print("Model inference time in total: ", total_time)
    print("Model inference time in total: ", t, "Average:", t/len(out))
    out = np.array(out)
    print(out.shape)

    if pos1 == 0:
        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        xx = np.arange(1, seg+1)
        integral = [np.sum(cg[i-1:]) for i in xx]
        plt.plot(xx, cg)
        # plt.plot(xx, integral, label="integral")
        # plt.legend()
        plt.ylabel("Absolute values of gradients")
        plt.xlabel("Input sequence number")
        plt.ylim([0, max(cg)*1.1])
        plt.tight_layout()
        # plt.savefig(r"./figures/FNN_input_contribution_buffer{}_sr{}.svg".format(seg, sr))
        # plt.show()

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

    # # plot the training and validation loss
    # acc_file = "./checkpoints/TC_FNN_45ca_threeTypeDecay_seg50_sr100_rsFalse_lr1e-3/TC_FNN_L2_acc_bs16_epoch100.pkl"
    # acc = pickle.load(open(acc_file, "rb"))
    # plt.figure(figsize=(20, 10))
    # plt.plot(acc['train'], '-', label='train')
    # plt.plot(acc['test'], '-', label='test')
    # plt.xlabel('Epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # # plt.savefig('./results/acc_{}.jpg')
    # # plt.show()
