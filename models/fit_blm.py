import os

import numpy as np
from backlash_model import BacklashModel
from matplotlib import pyplot as plt

from scipy.signal import butter
class OnlineSecondOrderFilter:
    def __init__(self, b, a, x, y):
        self.b = b  # Numerator coefficients [b0, b1, b2]
        self.a = a  # Denominator coefficients [a0, a1, a2], a0 should be 1
        self.x = [x, x, x]  # Initialize past inputs
        self.y = [y, y, y]  # Initialize past outputs

    def update(self, new_input):
        new_input = new_input[0]
        # Calculate new output
        new_output = self.b[0] * new_input + self.b[1] * self.x[1] + self.b[2] * self.x[2] - self.a[1] * self.y[1] - self.a[2] * self.y[2]

        # Update states
        self.x = [new_input, self.x[0], self.x[1]]
        self.y = [new_output, self.y[0], self.y[1]]

        return new_output

def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

def load_data(data_path, sample_rate=100, rm_init=1/7):
    data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    time = data[:, 0]
    tendon_disp = data[:, 1]
    tip_A = data[:, 8]

    # resample data using fixed sampling rate
    # sample_rate = 100  # Hz
    frequency = round(1 / (time[-1] / 7), 2)
    interp_time = np.arange(10, time[-1], 1 / sample_rate)
    tendon_disp = np.interp(interp_time, time, tendon_disp)
    tip_A = np.interp(interp_time, time, tip_A)
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency

    rm_init_number = int(rm_init * tip_A.shape[0])
    tendon_disp = tendon_disp[rm_init_number:]
    tip_A = tip_A[rm_init_number:]
    interp_time = interp_time[rm_init_number:]

    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'freq': freq}

def load_data_sets(dir="../tendon_data/innerTSlackSinus/train", sample_rate=25, rm_init=0):
    tendon_disp_all = []
    tip_A_all = []
    for file in os.listdir(dir):
        freq = float(os.path.basename(file).split("_")[1][:3])
        data_path = os.path.join(dir, file)
        data = load_data(data_path, sample_rate=sample_rate*freq*10, rm_init=rm_init)
        tendon_disp_all.append(data["tendon_disp"])
        tip_A_all.append(data["tip_A"])
    tendon_disp_all = np.concatenate(tendon_disp_all, axis=0)
    tip_A_all = np.concatenate(tip_A_all, axis=0)
    print(tendon_disp_all.shape)
    return {'tendon_disp': tendon_disp_all, 'tip_A': tip_A_all}

def save_backlash(data_path, out_path, data, predicted_tip):
    ori_data = np.loadtxt(data_path, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    predicted_tip_rs = np.interp(ori_data[:, 0], data["time"], predicted_tip[:, 0])
    label = ["Time", "Tendon1 Disp.", "Tendon2 Disp.", "Motor1 Curr.", "Motor2 Curr.", "Pos x", "Pos y", "Pos z",
             "Angle x", "Angle y", "Angle z", "backlash_tip"]
    data_to_save = np.hstack([ori_data, predicted_tip_rs[:, np.newaxis]])
    data_to_save = np.row_stack((label, data_to_save))
    np.savetxt(out_path, data_to_save, delimiter=',', fmt='%s')
    print(data_to_save.shape)

def fit_tendon(model_param: dict, model="forward"):
    # Setup model
    bl_model = BacklashModel(**model_param, x_init=np.array(0.0))
    rm_init = 0
    sr = 5
    # load data
    # data_path = r"../../hysteresis-fnn-lstm_ISMR\tendon_data\innerTSlack40\train\MidBL_01Hz_1.txt"
    # data = load_data(data_path, sample_rate=sr, rm_init=rm_init)
    data = load_data_sets(dir="../tendon_data/Data_with_Initialization/Sinus/train", sample_rate=sr, rm_init=rm_init)
    mean_tend = np.array(3) #np.mean(data["tendon_disp"])
    mean_tip = np.array(55)  #np.mean(data["tip_A"])
    print(mean_tend, mean_tip)
    tendon_disp = (data["tendon_disp"] - mean_tend)/1
    tip_A = (data["tip_A"] - mean_tip)/1

    x_hist = np.expand_dims(tendon_disp, axis=1)
    x_bl_hist = np.expand_dims(tip_A, axis=1)

    # print(x_hist.shape, x_bl_hist.shape)

    # Fit the model parameters
    bl_model.fit(u=x_hist[1:], x=x_bl_hist[1:], x_prev=x_bl_hist[:-1])

    test_data = "MidBL_0.45Hz"
    data_path = r"../tendon_data/Data_with_Initialization/Sinus/validate/{}.txt".format(test_data)
    data = load_data(data_path, sample_rate=25, rm_init=0)
    tendon_disp = (data["tendon_disp"] - mean_tend)/1
    tip_A = (data["tip_A"] - mean_tip)/1
    time = data["time"]
    x_ = np.expand_dims(tendon_disp, axis=1)
    y_ = np.expand_dims(tip_A, axis=1)

    # predict forward kinematics
    y_list = []
    for i in range(x_.shape[0]):
        x = x_[i, 0]
        y = bl_model(x)
        y_list.append(y.copy())
    # save to backlash prediction into txt file
    # out_path = r"../tendon_data/backlash_prediction/{}.txt".format(test_data)
    # save_backlash(data_path, out_path, data, y_list+mean_tip)

    print(bl_model.m_up, bl_model.m_lo, bl_model.c_lo, bl_model.c_up)
    # predict inverse kinematics
    x_list = []
    bl_model.u_pre = [x_[0, 0]]
    bl_model._x_prev = [y_[0, 0]]
    for i in range(y_.shape[0]):
        y = y_[i, 0]
        x = bl_model.inverse(y)
        x_list.append(x)

    # predict inverse kinematics with filter
    order = 2
    cutoff_frequency = 0.35  # Normalized frequency (cutoff frequency / (0.5 * sample_rate))
    # Generate filter coefficients for a low-pass filter
    b, a = butter(N=order, Wn=cutoff_frequency, btype='lowpass', analog=False)
    print(b, a)
    # Create filter instance
    # filter_instance = OnlineSecondOrderFilter(b, a, x_[0, 0], x_[0, 0])
    filter_instance = OnlineSecondOrderFilter(b, a, bl_model.u_s_pre, bl_model.u_s_pre)
    x_filter_list = []
    bl_model.u_pre = [x_[0, 0]]
    bl_model._x_prev = [y_[0, 0]]
    for i in range(y_.shape[0]):
        y = y_[i, 0]
        # x = bl_model.inverse(y)
        # x_filter_list.append(filter_instance.update(x))
        u_s = bl_model.update_u_s(y)
        new_u_s = filter_instance.update(u_s)
        x = bl_model.inverse_filtered(y, new_u_s)
        x_filter_list.append(x)

    # predict inverse kinematics with exponential
    bl_model.m_pre = bl_model.m_up
    x_exp_list = []
    bl_model.u_pre = [x_[0, 0]]
    bl_model._x_prev = [y_[0, 0]]
    for i in range(y_.shape[0]):
        y = y_[i, 0]
        x = bl_model.inverse_exponential(y, k=0.5, delta_t=1/sr)
        x_exp_list.append(x)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 * 1.5))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    if model == "forward":
        plt.plot(time, y_+mean_tip, "-", linewidth=0.8, label="Measured")
        plt.plot(time, y_list+mean_tip, linewidth=0.8, label="Backlash model")
        plt.xlabel("Time (S)", fontsize=8, labelpad=0.01)
        plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
        plt.title("RMSE:{:.3f} (deg)".format(rmse_norm(y_, y_list)), fontsize=8)
    else:
        plt.plot(time, x_ + +mean_tend, "-", linewidth=0.8, label="Measured")
        plt.plot(time, x_list + +mean_tend, linewidth=0.8, label="Backlash model")
        plt.plot(time, x_filter_list + +mean_tend, linewidth=0.8, label="Backlash filtered")
        # plt.plot(time, x_exp_list + np.mean(data["tendon_disp"]), linewidth=0.8, label="Backlash filtered")
        plt.xlim([20, 30])
        plt.xlabel("Time (S)", fontsize=8, labelpad=0.01)
        plt.ylabel("Tendon disp (mm)", fontsize=8, labelpad=0.01)
        plt.title("RMSE:{:.3f} (mm)".format(rmse_norm(x_, x_list)), fontsize=8)
    plt.legend(fontsize=8, frameon=False)

    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    if model == "forward":
        plt.plot(x_+mean_tend, y_+mean_tip, "-", linewidth=0.8, label="Measured")
        plt.plot(x_+mean_tend, y_list+mean_tip, linewidth=0.8, label="Backlash model")
        plt.xlabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
        plt.ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
        plt.ylim([15, 90])
        plt.xlim([-0.1, 6.1])
    else:
        plt.plot(y_ + mean_tip, x_ + mean_tend, "-", linewidth=0.8, label="Measured")
        plt.plot(y_ + mean_tip, x_list + mean_tend, linewidth=0.8, label="Backlash model")
        plt.plot(y_ + mean_tip, x_filter_list + mean_tend, linewidth=0.8, label="Backlash filtered")
        # plt.plot(y_ + np.mean(data["tip_A"]), x_exp_list + np.mean(data["tendon_disp"]), linewidth=0.8, label="Backlash filtered")
        plt.ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
        plt.xlabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(r"../../hysteresis-fnn-lstm_ISMR/figures/backlash model/{}_backlash_{}_{}.svg".format(model, test_data, rm_init))
    plt.savefig(r"../../hysteresis-fnn-lstm_ISMR/figures/backlash model/{}_backlash_{}_{}.png".format(model, test_data, rm_init), dpi=600)
    plt.show()

if __name__ == "__main__":
    # test_all(dict(m_lo=2.0, m_up=1.9, c_lo=1, c_up=1), True)
    # fit_tendon(dict(m_lo=2.0, m_up=1.9, c_lo=1, c_up=1))

    fit_tendon(dict(m_lo=2.0, m_up=1.9, c_lo=1, c_up=1), "forward")