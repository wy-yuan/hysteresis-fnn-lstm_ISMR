import os
import numpy as np
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
    tendon_disp = np.hstack([np.ones(seg)*0, tendon_disp_resample ]) #6
    tip_A = np.hstack([np.ones(seg)*30, tip_A_resample ]) #90
    tip_D = tip_disp_resample

    return {'time': interp_time, 'tendon_disp': tendon_disp, 'tip_A': tip_A, 'tip_D': tip_D}

# def load_data_alignWithTd(data_path, sample_rate=25):
#     data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
#     time = data[:, 0]
#     tendon_disp = data[:, 1]
#     tip_A = data[:, 8]
#
#     interp_td = np.arange(10, time[-1], 1 / sample_rate)
#     tip_A_resample = np.interp(interp_td, tendon_disp, tip_A)
#
#     td = np.arange(0, 6, 0.01)
#     tip_A_up = np.zeros_like(td)
#     tip_A_down = np.zeros_like(td)
#
#     pre_tip_A = tip_A[0]
#     pre_td = tendon_disp[0]
#     while i < len(tip_A):
#         i = i + 1
#         current_tip_A = tip_A[i]
#         current_td = tendon_disp[i]
#         while current_tip_A>pre_tip_A:
#             tip_A_up[index] = (current_tip_A + tip_A_up[index])
#
#             pre_tip_A = current_tip_A
#             i = i + 1
#             current_tip_A = tip_A[i]
#             current_td = tendon_disp[i]
#
#     return tip_A_up, tip_A_down

training_data = "./tendon_data/Data_with_Initialization/tip_pisition_ori/Stair5s"
testing_data = "./tendon_data/Data_with_Initialization/tip_pisition_ori/Stair"

tendon_disp_all = np.array([])
tip_A_all = np.array([])

for file in os.listdir(testing_data):
    path = os.path.join(testing_data, file)
    motion_name = os.path.basename(path).split(".txt")[0]
    data_path = path
    # data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    data = load_data(data_path, 0, sample_rate=1000)
    tendon_disp = data["tendon_disp"]
    tip_A = data["tip_A"]
    tendon_disp_all = np.hstack([tendon_disp_all, tendon_disp])
    tip_A_all = np.hstack([tip_A_all, tip_A])

    plt.plot(tendon_disp, tip_A, "b", label="testing data", alpha=0.1)

res = 0.1
# calculate mean value of tip angle
td = np.arange(0, 6+res, res)
tip_A_avg = np.zeros_like(td)
for i in range(len(td)):
    index = np.where( (tendon_disp_all >= (td[i]-res/2)) * (tendon_disp_all < (td[i]+res/2)) )
    tip_A_avg[i] = np.mean(tip_A_all[index])
plt.plot(td, tip_A_avg, "b")


tendon_disp_all = np.array([])
tip_A_all = np.array([])

for file in os.listdir(training_data):
    path = os.path.join(training_data, file)
    motion_name = os.path.basename(path).split(".txt")[0]
    data_path = path
    # data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    data = load_data(data_path, 0, sample_rate=1000)
    tendon_disp = data["tendon_disp"]
    tip_A = data["tip_A"]
    tendon_disp_all = np.hstack([tendon_disp_all, tendon_disp])
    tip_A_all = np.hstack([tip_A_all, tip_A])

    plt.plot(tendon_disp, tip_A, "r", label="training data", alpha=0.1)

# calculate mean value of tip angle
td = np.arange(0, 6+res, res)
tip_A_avg = np.zeros_like(td)
for i in range(len(td)):
    index = np.where( (tendon_disp_all >= (td[i]-res/2)) * (tendon_disp_all < (td[i]+res/2)) )
    tip_A_avg[i] = np.mean(tip_A_all[index])
plt.plot(td, tip_A_avg, "r")

plt.xlabel("tendon displacement (mm)")
plt.ylabel("tip angle (degrees)")
plt.show()