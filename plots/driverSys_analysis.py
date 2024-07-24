import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append("..")
from generate_random_motion import generate_withDuration
def load_data(data_path, sample_rate=100, rm_init=1/7):
    data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    time = data[:, 0]
    tendon_disp = data[:, 1]
    tip_A = data[:, 8]

    # resample data using fixed sampling rate
    # sample_rate = 100  # Hz
    frequency = round(1 / (time[-1] / 7), 2)
    interp_time = np.arange(time[0], time[-1], 1 / sample_rate)
    tendon_disp = np.interp(interp_time, time, tendon_disp)
    tip_A = np.interp(interp_time, time, tip_A)
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency

    rm_init_number = int(rm_init * tip_A.shape[0])
    tendon_disp = tendon_disp[rm_init_number:]
    tip_A = tip_A[rm_init_number:]
    interp_time = interp_time[rm_init_number:]

    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'freq': freq}

def sin_input_signal(x, f, type="non_decay"):
    n = 1000
    A = 6/2
    tau = (-f / 6) * np.log(2 / 14)   # f/6*log(7)
    # y = A*np.exp(-tau*x)*(np.sin(2*np.pi*f*x-np.pi/2)+1)

    # non decaying
    if type == "non_decay":
        y = A * (np.sin(2 * np.pi * f * x - np.pi / 2) + 1)
        dy = A * 2 * np.pi * f * np.cos(2 * np.pi * f * x - np.pi / 2)

    # # one direction decaying (0 base-line)
    if type == "0bl":
        y = A * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2) + 1)
        dy = -A * tau * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2) + 1) + A * np.exp(
            -tau * x) * 2 * np.pi * f * np.cos(2 * np.pi * f * x - np.pi / 2)
    #
    # # one direction decaying (45 base-line)
    if type == "midbl":
        y = A * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2)) + A
        dy = -A * tau * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2)) + A * np.exp(
            -tau * x) * 2 * np.pi * f * np.cos(2 * np.pi * f * x - np.pi / 2)
    #
    # # one direction decaying (90 base-line)
    if type == "endbl":
        y = A * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2) - 1) + 2*A
        dy = -A * tau * np.exp(-tau * x) * (np.sin(2 * np.pi * f * x - np.pi / 2) - 1) + A * np.exp(
            -tau * x) * 2 * np.pi * f * np.cos(2 * np.pi * f * x - np.pi / 2)

    return y



if __name__ == '__main__':
    sr = 25
    test_data = "EndBL_01Hz_1"
    freq = 0.1
    data_path = r"../tendon_data/innerTSlack40/train/{}.txt".format(test_data)

    test_data = "step_motion2_10"
    data_path = r"../tendon_data/innerTSlack40/test/{}.txt".format(test_data)
    data = load_data(data_path, sample_rate=sr, rm_init=0)
    time = data['time']
    measured = data['tendon_disp']

    # desired tendon displacement
    # desired = sin_input_signal(time, f=freq, type="endbl")
    long_pause = [[0, 3, 6, 3, 0],
                   [20, 20, 20, 20, 20]]
    max_time = 100
    x_new, y_new, y_de = generate_withDuration(long_pause, max_time, window_size=30)
    desired = np.interp(time, x_new, y_new)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 * 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(time, desired, linewidth=0.8, label='Desired')
    plt.plot(time, measured, linewidth=0.8, label='Measured')
    plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
    plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
    plt.legend(fontsize=8, frameon=False, loc='upper right', ncol=2)

    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(desired, measured, linewidth=0.8)
    plt.xlim([-0.1, 6])
    plt.ylim([-0.1, 6])
    plt.xlabel('Desired Tendon disp. (mm)', fontsize=8, labelpad=0.01)
    plt.ylabel('Measured Tendon disp. (mm)', fontsize=8, labelpad=0.01)

    plt.tight_layout()
    plt.savefig(r"../figures/driverSys_analysis/{}.svg".format(test_data))
    plt.savefig(r"../figures/driverSys_analysis/{}.png".format(test_data), dpi=600)
    plt.show()