import os
import numpy as np
import matplotlib.pyplot as plt

# tensor 2 position [7.98, 1.13, -1.21]
folder_list = [
                "../tendon_data/Data_with_Initialization/tip_pisition_ori/Stair5s",
               "../tendon_data/Data_with_Initialization/tip_pisition_ori/Stair",
               "../tendon_data/Data_with_Initialization/tip_pisition_ori/ramp",
               "../tendon_data/Data_with_Initialization/tip_pisition_ori/staircase",
               "../tendon_data/Data_with_Initialization/tip_pisition_ori/Sinus",
               "../tendon_data/Data_with_Initialization/tip_pisition_ori/random",
               "../tendon_data/Data_with_Initialization/tip_pisition_ori/sinus_stationary"
               ]

plt.figure(figsize=(88.9 / 25.4/2, 88.9 / 25.4/2))
plt.rcParams['font.family'] = 'Times New Roman'
plt.tick_params(labelsize=8, pad=0.1, length=2)

angle = []
X_ = []
Y_ = []
index = 0
for folder_path in folder_list:
    # folder_path = "../tendon_data/Data_with_Initialization/tip_pisition_ori/Stair"
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        motion_name = os.path.basename(path).split(".txt")[0]
        data_path = path

        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        tip_A = data[:, 8]
        X = 7.98*25.4 - data[:, 5] + 14  # unit mm
        Y = data[:, 6] - 1.13*25.4  # unit mm
        # x_color = "lightblue"
        # y_color = "pink"
        x_color = "blue"
        y_color = "red"
        if index == 0:
            plt.plot(tip_A, X, x_color, linewidth=0.8, label="Measured x")
            plt.plot(tip_A, Y, y_color, linewidth=0.8, label="Measured y")
            index = 1
        else:
            plt.plot(tip_A, X, x_color, linewidth=0.8)
            plt.plot(tip_A, Y, y_color, linewidth=0.8)
        angle = np.hstack((angle, tip_A))
        X_ = np.hstack((X_, X))
        Y_ = np.hstack((Y_, Y))
# cc_dist_X = 70*((np.sin(tip_A*np.pi/180))/(tip_A*np.pi/180))  # use constant curvature model to calculate distance
# plt.plot(tip_A, cc_dist_X, 'b')
# cc_dist_Y = 70*((1-np.cos(tip_A*np.pi/180))/(tip_A*np.pi/180))  # use constant curvature model to calculate distance
# plt.plot(tip_A, cc_dist_Y, 'r')
coef_x = np.polyfit(angle, X_, 2)
print("coef_x:", coef_x)
p_x = np.poly1d(coef_x)
coef_y = np.polyfit(angle, Y_, 2)
print("coef_y:", coef_y)
p_y = np.poly1d(coef_y)

# coef_x: [-3.62136415e-03 -4.61111115e-02  7.04893736e+01]
# coef_y: [-0.00326109  0.77266619  0.21515682]

a = np.linspace(0, 90, 100)

# plt.plot(a, p_x(a), 'b', linewidth=0.8, label="Fitted x")
# plt.plot(a, p_y(a), 'r', linewidth=0.8, label="Fitted y")
plt.xlim([0, 90])
plt.xticks([0, 30, 60, 90])
plt.ylim([0, 90])
plt.yticks([0, 30, 60, 90])
plt.xlabel("Tip angle (degree)", fontsize=8)
plt.ylabel("Tip position (mm)", fontsize=8)
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("../figures/RAL_plots/Ang_TipPos_XY.svg")
# plt.savefig("../figures/Data_with_initialization_SinusStair5s/Ang_TipPos_XY.png", dpi=600)
plt.show()

plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4))
plt.rcParams['font.family'] = 'Times New Roman'
plt.tick_params(labelsize=10, pad=0.1, length=2)
index = 0
index2 = 0
for folder_path in folder_list:
    # folder_path = "../tendon_data/Data_with_Initialization/tip_pisition_ori/Stair"
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        motion_name = os.path.basename(path).split(".txt")[0]
        data_path = path

        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        tendon_disp = data[:, 1]
        tip_A = data[:, 8]
        X = 7.98*25.4 - data[:, 5] + 14  # unit mm
        Y = data[:, 6] - 1.13*25.4  # unit mm
        x_color = "b"
        y_color = "r"
        if index == 0:
            plt.plot(tendon_disp, X, x_color, linewidth=0.8, label="Measured X")
            plt.plot(tendon_disp, Y, y_color, linewidth=0.8, label="Measured X")
            index = 1
        else:
            plt.plot(tendon_disp, X, x_color, linewidth=0.8)
            plt.plot(tendon_disp, Y, y_color, linewidth=0.8)
        # plt.plot(tendon_disp, tip_A, linewidth=0.8)

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        motion_name = os.path.basename(path).split(".txt")[0]
        data_path = path

        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        tendon_disp = data[:, 1]
        tip_A = data[:, 8]
        X = 7.98*25.4 - data[:, 5] + 14  # unit mm
        Y = data[:, 6] - 1.13*25.4  # unit mm
        x_color = "lightblue"
        y_color = "pink"
        if index2 == 0:
            plt.plot(tendon_disp, p_x(tip_A), x_color, linewidth=0.8, label="Mapped X")
            plt.plot(tendon_disp, p_y(tip_A), y_color, linewidth=0.8, label="Mapped Y")
            index2 = 1
        else:
            plt.plot(tendon_disp, p_x(tip_A), x_color, linewidth=0.8)
            plt.plot(tendon_disp, p_y(tip_A), y_color, linewidth=0.8)
plt.ylim([0, 90])
plt.xlabel("Tip angle (deg)")
plt.ylabel("Tip position (mm)")
plt.legend(fontsize=8, frameon=False)
# plt.savefig("../figures/Data_with_initialization_SinusStair5s/tendon_disp_Ang_TipPos_XY.png", dpi=600)
plt.show()