import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    tendon_disp = np.hstack([np.ones(seg)*0, tendon_disp_resample]) #6
    tip_A = np.hstack([np.ones(seg), tip_A_resample]) #90
    tip_D = np.vstack([np.hstack([np.ones((seg,1))*66/90, np.ones((seg,1))*20]), tip_disp_resample ])
    freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'tip_D': tip_D, 'freq': freq}

if __name__ == '__main__':
    folder_path = r"D:\yuan\Dropbox (BCH)\bch\2024_ISMR\presentation\three_offsets"
    plt.figure(figsize=(88.9 / 25.4 , 88.9 / 25.4 * 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=12, pad=0.01, length=2)
    for file in os.listdir(folder_path):
        if file.split(".")[-1]=="png":
            continue
        path = os.path.join(folder_path, file)
        data = load_data(path, 0, sample_rate=25)


        plt.plot(data["time"]-10, data["tendon_disp"], linewidth=1)
        plt.xlabel("Time (s)", fontsize=12, labelpad=0.01)
        plt.ylabel("Tendon disp. (mm)", fontsize=12, labelpad=0.01)
        plt.xlim([0, 15])

    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=12, pad=0.01, length=2)
    plt.plot(data["tendon_disp"], data["tip_A"], linewidth=1)
    plt.xlabel("Tendon disp. (mm)", fontsize=12, labelpad=0.01)
    plt.ylabel("Tip Rotation θ (deg)", fontsize=12, labelpad=0.01)
    plt.xlim([-0.1, 6.1])
    plt.ylim([20, 90])

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "three_offsets.png"), dpi=600)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=12, pad=0.01, length=2)
    plt.plot(data["tendon_disp"], data["tip_A"], linewidth=1)
    plt.xlabel("Tendon displacement (mm)", fontsize=12, labelpad=0.01)
    plt.ylabel("Tip Rotation θ (deg)", fontsize=12, labelpad=0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "three_offsets_theta.png"), dpi=600)

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4/1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=12, pad=0.01, length=2)
    plt.plot(data["tendon_disp"], data["tip_D"][:,0], linewidth=1, label="X")
    plt.plot(data["tendon_disp"], data["tip_D"][:,1], linewidth=1, label="Y")
    plt.xlabel("Tendon displacement (mm)", fontsize=12, labelpad=0.01)
    plt.ylabel("Tip position X, Y (mm)", fontsize=12, labelpad=0.01)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "three_offsets_XY.png"), dpi=600)
    # plt.show()

    # calculate total data points
    folder_path = r"../tendon_data/Data_with_Initialization/SinusStair5s/Sinus"
    sum = 0
    for file in os.listdir(folder_path):
        if file.split(".")[-1] == "png":
            continue
        path = os.path.join(folder_path, file)
        data = load_data(path, 0, sample_rate=25)
        sum += len(data["time"])
    print(sum)
