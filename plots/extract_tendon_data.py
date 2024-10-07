import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math

def sin_input(x, f):
    A = 6/2
    tau = (-f / 6) * np.log(2 / 14)

    # both direction non-decay
    y = A * np.sin(2 * np.pi * f * x)
    dy = A * 2 * np.pi * f * np.cos(2 * np.pi * f * x)
    return y, dy

def generate_withDuration(long_pause, max_time, window_size = 30):
    sig = long_pause[0]
    repeats = long_pause[1]
    y_true = np.array([])
    for s,r in zip(sig, repeats):
        y_true = np.hstack([y_true, np.ones(r)*s])

    x = np.linspace(0, max_time, len(y_true))
    x_new = np.linspace(0, max_time, 1000)
    # x_new_ = np.linspace(0, max_time, 1000+window_size-1)
    # y_new = np.interp(x_new_, x, y_true)
    # smooth_window = np.ones(window_size) / window_size
    # y_new = np.convolve(y_new, smooth_window, mode='valid')
    y_new = np.interp(x_new, x, y_true)
    smooth_window = np.ones(window_size) / window_size
    y_smooth = np.ones_like(y_new)
    for i in range(y_new.shape[0]):
        sum = 0.0
        for j in range(window_size):
            if i - j >= 0:
                sum += y_new[i - j] * smooth_window[j]
        y_smooth[i] = sum
    y_new = y_smooth
    # y_new = y_new - y_new[0]  #  make tendon displacement start from 0
    # y_new = (y_new - y_new[0])*12+20  #  make tendon displacement start from 0
    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    y_de = np.hstack([np.zeros(1), y_de])
    print(y_de.shape)
    return x_new, y_new, y_de

def generate_ramp_trail(x, y_true, max_time):
    # pos_number = np.random.randint(5, 10, 1)[0]
    # position = np.random.rand(pos_number) * 6
    # interval = np.random.rand(pos_number)*3+1
    # interval_time = interval/sum(interval)*max_time
    #
    # x = np.hstack([0, np.around(np.cumsum(interval_time), decimals=2)])
    # y_true = np.hstack([0, position])
    x_new = np.arange(0, max_time, 0.01)
    y_new = np.interp(x_new, x, y_true)

    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    y_de = np.hstack([np.zeros(1), y_de])

    return x_new, y_new, y_de

def cal_rotation(v1, normal_vector=[-0.68784265, -0.05311392, 0.72391271]): #[-0.76435347,-0.13500009,0.61676366]
    n = normal_vector / np.linalg.norm(normal_vector)
    v0 = np.array([0, 1, 0])
    v0_proj = v0 - np.dot(v0, n) * n

    v1_proj = v1 - np.dot(v1, n) * n
    cos_theta = np.dot(v0_proj, v1_proj) / (np.linalg.norm(v0_proj) * np.linalg.norm(v1_proj))
    theta = np.arccos(cos_theta)
    cross_product = np.cross(v0_proj, v1_proj)
    # Determine the sign of the angle
    if np.dot(cross_product, n) > 0:
        theta = -theta
    theta_degrees = np.degrees(theta)
    return theta_degrees

# Define constants
color = np.array([
    [0, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840]
])
font_size = 50
lbs = 50
baseline = 0
# directory = r"D:/yuan/Dropbox (BCH)/bch/our tendon/0_1 Hz_copy"
directory = r"D:\yuan\Dropbox (BCH)\LearningDatasets\Data_innerTendonSlack\2024-03-13\random_staircase"
directory = r"D:\yuan\BCH Dropbox\Wang Yuan\LearningDatasets\InnerSheath2024\0923"
freq = 0.5
data_prefix = "pre_"
downfirst = 0
# base_pos = np.array([5.59, 1.068, -0.62])  #y(1.068) coordinate of sensor2,  x(5.59) and z(-0.62) is estimated from sensor1 position
# base_pos = np.array([5.76, 1.15, -0.60])  #y(1.068) coordinate of sensor2,  x(5.59) and z(-0.62) is estimated from sensor1 position
base_pos = np.array([0, 0, 0])
# sensor 2 position [7.98, 1.13, -1.21]
# base_pos = np.array([0, 0, 0])  #y(1.068) coordinate of sensor2,  x(5.59) and z(-0.62) is estimated from sensor1 position
# base_pos = np.array([5, -5, 5])
motors = [3, 4]   # numbers start from 0

# Open files
dataset = os.listdir(directory)
print(dataset)
label = ["Time", "Tendon1 Disp.", "Tendon2 Disp.", "Motor1 Curr.", "Motor2 Curr.", "Pos x", "Pos y", "Pos z", "Angle x", "Angle y", "Angle z"]
colorIndex = 0

# Define data storage variables
em_data = {}
rosData = {}
ros_interp_time = None

# Extract all data
for item in dataset:
    print(item)
    if "rosbag" not in item:
        continue

    folders = os.listdir(os.path.join(directory, item))
    folders = [folder for folder in folders if not folder.startswith('.')]

    for folder_name in folders:
        if os.path.isdir(os.path.join(directory, item, folder_name)):
            files = os.listdir(os.path.join(directory, item, folder_name))
            files = [file for file in files if file.endswith(".csv")]

        if len(folder_name.split(".")) > 1 or "Test" in folder_name:
            if "." in folder_name or "Test" in folder_name:
                name_split = folder_name.split(".")
                if name_split[-1] == "csv" or "Test" in folder_name:
                    # EM data processing
                    file_path = os.path.join(directory, item, folder_name)
                    if os.path.exists(file_path):  # Check if the file exists
                        if file_path.endswith('.csv') or "Test" in folder_name:
                            # Read the CSV file using pandas (assumes you have pandas installed)
                            try:
                                sheet = pd.read_table(file_path, delimiter=",", usecols=range(0, 31), index_col=None)
                                sheet.columns = ['timestamp', 'status', 'button', 'x1', 'y1', 'z1', 'a1', 'e1', 'r1',
                                                 'q1', 'x2', 'y2', 'z2', 'a2', 'e2', 'r2', 'q2', 'x3', 'y3', 'z3', 'a3',
                                                 'e3', 'r3', 'q3', 'x4', 'y4', 'z4', 'a4', 'e4', 'r4', 'q4']
                            except ValueError as e:
                                sheet = pd.read_table(file_path, sep=r"\s+", skiprows=1, header=None, index_col=None)
                                sheet.columns = ['sensor1', 'status1', 'x1', 'y1', 'z1', 'a1', 'e1', 'r1', 'button', 'q1', 'timestamp',
                                                 'sensor2', 'status2', 'x2', 'y2', 'z2', 'a2', 'e2', 'r2', 'button2', 'q2', 'timestamp2']
                            # print(sheet.shape)
                            # print(sheet)
                            # Time
                            print("------------------------------")
                            em_data['time'] = sheet['timestamp'] - sheet['timestamp'][0]
                            # Tip Sensor
                            em_data['sensor_tip'] = {}
                            em_data['sensor_tip']['pos'] = np.array([sheet['x1'], sheet['y1'], sheet['z1']])
                            em_data['sensor_tip']['or'] = np.array([sheet['a1'], sheet['e1'], sheet['r1']])
                            # Base Sensor
                            em_data['sensor_base'] = {}
                            em_data['sensor_base']['pos'] = np.array([sheet['x2'], sheet['y2'], sheet['z2']])
                            em_data['sensor_base']['or'] = np.array([sheet['a2'], sheet['e2'], sheet['r2']])

                            if isinstance(list(sheet["button"])[0], str):
                                # Status
                                em_data['status'] = (sheet['button'] == "B_1100")
                            else:
                                # Status
                                em_data['status'] = (sheet['button'] == 1)

            if folder_name == "README.txt":
                with open(os.path.join(directory, item, folder_name), 'r') as file:
                    titles = file.read().splitlines()
                    fig_title = titles[0]
            continue

        for file_name in files:
            var_name = file_name.split(".")
            if var_name[1] != "csv" or var_name[0] == "pause_data" or var_name[0] == "ramp_trail" or var_name[0] == "staircase_trail":
                continue

            if folder_name == "rosData":
                # ROS data
                sheet = pd.read_csv(os.path.join(directory, item, folder_name, file_name))
                if "data" in sheet.columns:
                    data_time = sheet['time']
                    data_data = sheet['data']
                else:
                    # print(sheet['header'])
                    # data_time = [float(x['secs']) + float(x['nsecs']) * 1e-9 for x in sheet['header']]
                    data_header = sheet['header']
                    data_time = []
                    for nsec_str in data_header:
                        matches = re.findall(r"(\d+)", nsec_str)
                        secs, nsecs = map(int, matches)
                        time = secs + nsecs * 1e-9
                        data_time.append(time)
                    # data_time = np.array(data_time) - data_time[0]
                    # data_time = data_time.tolist()

                    data = {}
                    for col in sheet.columns[3:]:
                        data[col] = []
                        for val in sheet[col]:
                            split = val.split(',')
                            # values = [float(s[1:-1]) if s != '[]' else np.nan for s in split]
                            # print(values)
                            values = []
                            for m in range(len(split)):
                                if split[m] == "[]":
                                    break
                                else:
                                    if m < len(split) - 1:
                                        values.append(float(split[m][1:]))
                                    else:
                                        values.append(float(split[m][1:-1]))

                            data[col].append(values)

                    rosData[var_name[0]] = {}
                    rosData[var_name[0]]['time'] = data_time
                    rosData[var_name[0]]['header'] = data_header
                    rosData[var_name[0]].update(data)
                    del data

    # Align data
    ros_interp_time_init = rosData['EM_trigger_running_time']['time']
    ros_interp_time = rosData['EM_trigger_running_time']['time']
    ros_joint_time = rosData['joint_data_feedback']['time']
    ros_command_time = rosData['joint_command_mm']['time']
    # ros_commandVel_time = rosData['joint_data_vel_command']['time']
    ros_commandVel_time = rosData['joint_data_pos_command']['time']  # for position control
    ros_emData_time = rosData['em_data']['time']  # em tracker data

    ros_command_mm = np.array(rosData['joint_command_mm']['velocity'])
    ros_command_mm = np.vstack([np.interp(ros_interp_time, ros_command_time, ros_command_mm[:, motors[0]]), np.interp(ros_interp_time, ros_command_time, ros_command_mm[:, motors[1]])]).T
    ros_command_mm = ros_command_mm[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    # ros_command_vel = np.array(rosData['joint_data_vel_command']['velocity'])
    ros_command_vel = np.array(rosData['joint_data_pos_command']['position'])  # for position control
    ros_command_vel = np.vstack([np.interp(ros_interp_time, ros_commandVel_time, ros_command_vel[:, motors[0]]), np.interp(ros_interp_time, ros_commandVel_time, ros_command_vel[:, motors[1]])]).T

    ros_command_effort = np.array(rosData['joint_data_pos_command']['effort'])

    ros_pwm_time = rosData['pwm_data_feedback']['time']
    ros_pwm = np.array(rosData['pwm_data_feedback']['effort'])
    ros_pwm_interp = np.vstack([np.interp(ros_interp_time, ros_pwm_time, ros_pwm[:, motors[0]]),
                               np.interp(ros_interp_time, ros_pwm_time, ros_pwm[:, motors[1]])]).T

    ros_joint_pos = -np.array(rosData['joint_data_feedback']['position'])
    ros_joint_pos = np.vstack([np.interp(ros_interp_time, ros_joint_time, ros_joint_pos[:, motors[0]]), np.interp(ros_interp_time, ros_joint_time, ros_joint_pos[:, motors[1]])]).T
    # ros_joint_pos = np.interp(ros_interp_time, ros_joint_time, ros_joint_pos[:, 0]).T
    ros_joint_pos = ros_joint_pos[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    ros_joint_pos -= ros_joint_pos[0]

    ros_joint_vel = np.array(rosData['joint_data_feedback']['velocity'])
    ros_joint_vel = np.vstack([np.interp(ros_interp_time, ros_joint_time, ros_joint_vel[:, motors[0]]), np.interp(ros_interp_time, ros_joint_time, ros_joint_vel[:, motors[1]])]).T
    ros_joint_vel = ros_joint_vel[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]

    ros_joint_eff = np.array(rosData['joint_data_feedback']['effort'])
    ros_joint_eff = np.vstack([np.interp(ros_interp_time, ros_joint_time, ros_joint_eff[:, motors[0]]), np.interp(ros_interp_time, ros_joint_time, ros_joint_eff[:, motors[1]])]).T
    ros_interp_time = np.array(ros_interp_time)
    ros_interp_time_0 = ros_interp_time[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] == 0]

    ros_joint_eff_0 = ros_joint_eff[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] == 0]  # motor current before experiment start
    ros_joint_eff = ros_joint_eff[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]


    ros_emData = np.array(rosData['em_data']['position'])
    EMn = 1
    ind = ros_emData[:, 6*EMn+3] > 0
    ind_ = ros_emData[:, 6*EMn+3] < 0
    ros_emData[ind, 6*EMn+3] = -180 + ros_emData[ind, 6*EMn+3]
    ros_emData[ind_, 6*EMn+3] = 180 + ros_emData[ind_, 6*EMn+3]
    emData_pos0 = np.vstack([np.interp(ros_interp_time, ros_emData_time, ros_emData[:, 6*EMn+0]), np.interp(ros_interp_time, ros_emData_time, ros_emData[:, 6*EMn+1]),
                             np.interp(ros_interp_time, ros_emData_time, ros_emData[:, 6*EMn+2])]).T
    print(emData_pos0.shape, (np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0).shape)
    emData_pos0 = emData_pos0[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    emData_rot0 = np.vstack([np.interp(ros_interp_time, ros_emData_time, ros_emData[:, 6*EMn+3]), np.interp(ros_interp_time, ros_emData_time, ros_emData[:, 6*EMn+4]),
                             np.interp(ros_interp_time, ros_emData_time, ros_emData[:, 6*EMn+5])]).T
    emData_rot_pretension = emData_rot0[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] == 0]
    emData_rot0 = emData_rot0[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]

    ros_interp_time = ros_interp_time[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    ros_interp_time_0 -= ros_interp_time[0]
    ros_interp_time_init -= ros_interp_time[0]
    ros_interp_time -= ros_interp_time[0]

    em_pos = emData_pos0[:, :] - base_pos[:, np.newaxis].T
    em_pos = 25.4 * em_pos
    em_rot = emData_rot0
    # ind = em_rot[:, 0] > 0
    # ind_ = em_rot[:, 0] < 0
    # em_rot[ind, 0] = -180 + em_rot[ind, 0]
    # em_rot[ind_, 0] = 180 + em_rot[ind_, 0]

    em_disp = np.linalg.norm(em_pos, axis=1)
    em_pos_interp = emData_pos0
    em_rot_interp = em_rot
    # em_disp = em_pos_interp[:, 0]
    # em_rot_interp -= em_rot_interp[0, :]    # make rotation align to the start point

    # Hysteresis
    f_num = 9
    current_lim = [-15, 65]
    for n in range(2):
        plt.figure(1+n*f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_pos[:, n], em_disp, linewidth=2.0)
        # slope, intercept = np.polyfit(em_rot_interp[:, 0], em_disp, 1)
        # print("slope:", slope)
        # plt.plot(em_rot_interp[:, 0], em_disp, linewidth=2.0)
        # plt.plot(em_rot_interp[:, 0], slope * em_rot_interp[:, 0] + intercept, "r", linewidth=2.0)
        plt.grid(True)
        plt.xlabel('Tendon{} Displacement (mm)'.format(n+1), fontsize=font_size)
        # plt.xlabel('Tip Rotation azimuth (deg)'.format(n+1), fontsize=font_size)
        plt.ylabel('Tip Displacement (mm)', fontsize=font_size)

        plt.figure(2+n*f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_pos[:, n], em_rot_interp[:, 0], linewidth=2.0)
        plt.grid(True)
        plt.ylim([-10, 40])
        plt.xlabel('Tendon Displacement (mm)', fontsize=font_size)
        plt.ylabel('Tip Azimuth (deg)', fontsize=font_size)

        plt.figure(23 + n, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_pos[:, n], em_rot_interp[:, 1], linewidth=2.0)
        plt.grid(True)
        plt.xlabel('Tendon{} Displacement (mm)'.format(n + 1), fontsize=font_size)
        plt.ylim([-10, 50])
        plt.ylabel('Tip Rotation elevation (deg)', fontsize=font_size)

        bending_angle = []
        for xx in range(em_rot_interp.shape[0]):
            azimuth = math.radians(em_rot_interp[xx, 0])  # Example azimuth angle 1
            elevation = math.radians(em_rot_interp[xx, 1])  # Example elevation angle 1
            # bending_radian = math.atan2(math.sin(azimuth) + math.sin(elevation), math.cos(azimuth) + math.cos(elevation))
            x = math.cos(elevation) * math.sin(azimuth)
            y = math.cos(elevation) * math.cos(azimuth)
            z = math.sin(elevation)
            # Reference direction (e.g., vertical direction)
            # reference_direction = [0, 1, 0]  # Vertical direction in Cartesian coordinates
            # # Calculate the dot product between the direction vector and the reference direction
            # dot_product = x * reference_direction[0] + y * reference_direction[1] + z * reference_direction[2]
            # # Calculate the bending angle in degrees
            # bending_radian = math.acos(dot_product)
            # bending_angle.append(math.degrees(bending_radian))
            v1 = np.array([x, y, z])
            theta_degrees = cal_rotation(v1)
            bending_angle.append(theta_degrees)
        plt.figure(3 + n * f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_pos[:, n], bending_angle, linewidth=2.0)
        plt.grid(True)
        plt.xlabel('Tendon{} Displacement (mm)'.format(n + 1), fontsize=font_size)
        plt.ylim([-10, 60])
        plt.ylabel('Tip Rotation (deg)', fontsize=font_size)

        plt.figure(4+n*f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_eff[:, n], em_disp, linewidth=2.0)
        plt.grid(True)
        # plt.xlim(current_lim)
        plt.xlabel('Motor{} Current (mA)'.format(n+1), fontsize=font_size)
        plt.ylabel('Tip Displacement (mm)', fontsize=font_size)

        plt.figure(5+n*f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_eff[:, n], em_rot_interp[:, 0], linewidth=5.0)
        plt.grid(True)
        plt.ylim([10, 90])
        plt.xlabel('Motor Current (mA)', fontsize=font_size)
        plt.ylabel('Tip Rotation (deg)', fontsize=font_size)

        plt.figure(6 + n * f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        plt.plot(ros_joint_eff[:, n], ros_joint_pos[:, n], "-", linewidth=2.0)
        # plt.plot(ros_joint_eff[:1, n], ros_joint_pos[:1, n], "o", linewidth=2.0)
        plt.grid(True)
        # plt.xlim(current_lim)
        plt.xlabel('Motor{} Current (mA)'.format(n + 1), fontsize=font_size)
        plt.ylabel('Tendon{} Displacement (mm)'.format(n+1), fontsize=font_size)

        plt.figure(7 + n * f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        inx_0s = ros_interp_time_0<=0
        inx_end = ros_interp_time_0>0
        inx_10s = ros_interp_time<=10
        # plt.plot(np.hstack([ros_interp_time_0[inx_0s], ros_interp_time[inx_10s]]), np.hstack([ros_joint_eff_0[inx_0s, n], ros_joint_eff[inx_10s, n]]), "-", linewidth=2.0)
        plt.plot(np.hstack([ros_interp_time_0[inx_0s], ros_interp_time, ros_interp_time_0[inx_end]]), np.hstack([ros_joint_eff_0[inx_0s, n], ros_joint_eff[:, n], ros_joint_eff_0[inx_end, n]]), "-", linewidth=2.0)
        plt.grid(True)
        plt.xlabel('Time (s)', fontsize=font_size)
        plt.ylabel('Motor{} Current (mA)'.format(n + 1), fontsize=font_size)
        # plt.ylim(current_lim)

        plt.figure(8 + n * f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        inx_0s = ros_interp_time_0 <= 0
        inx_end = ros_interp_time_0 > 0
        inx_10s = ros_interp_time <= 10
        plt.plot(np.hstack([ros_interp_time_0[inx_0s], ros_interp_time, ros_interp_time_0[inx_end]]), ros_command_vel[:, n], "-", linewidth=2.0)
        plt.grid(True)
        # plt.ylim([-1, 1])
        # plt.xlim([-20, 5])
        plt.xlabel('Time (s)', fontsize=font_size)
        plt.ylabel('Motor{} command (counts)'.format(n + 1), fontsize=font_size)

        plt.figure(9 + n * f_num, figsize=(20, 15))
        plt.tick_params(labelsize=lbs)
        # slope, intercept = np.polyfit(em_rot_interp[:, 0], em_disp, 1)
        # print("slope:", slope)
        # plt.plot(em_rot_interp[:, 0], em_pos_interp[:, 0], "r", linewidth=2.0)

        ## ---- plot tip position vs. tip angle, and compare it with Constant Curvature Model
        # if item.split("_")[-1] == "2":
        #     plt.plot(em_rot_interp[:, 0], em_pos_interp[:, 1], "b", linewidth=1.0)
        #     cc_dist = 71*((1-np.cos(em_rot_interp[:, 0]*np.pi/180))/(em_rot_interp[:, 0]*np.pi/180))  # use constant curvature model to calculate distance
        #     plt.plot(em_rot_interp[:, 0], cc_dist, "r", linewidth=2.0)

        # plt.plot(em_rot_interp[:, 0], slope * em_rot_interp[:, 0] + intercept, "r", linewidth=2.0)
        plt.grid(True)
        plt.xlabel('Tip Rotation azimuth (deg)'.format(n+1), fontsize=font_size)
        plt.ylabel('Tip Distance (mm)', fontsize=font_size)

    # fig = plt.figure(figsize=(20, 20))
    # ax = fig.add_subplot(projection='3d')
    # em_pos_interp -= em_pos_interp[0,:]
    # ax.plot(em_pos_interp[:, 0], em_pos_interp[:, 1], em_pos_interp[:, 2], marker="", linewidth=2)
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    plt.figure(19, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    # em_pos_interp -= em_pos_interp[0, :]
    plt.plot(em_pos_interp[:, 1], -em_pos_interp[:, 2], "-", linewidth=2.0)
    plt.grid(True)
    plt.xlim([-35, 35])
    plt.ylim([-30, 30])
    plt.xlabel('Y', fontsize=font_size)
    plt.ylabel('Z', fontsize=font_size)

    plt.figure(20, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    y, dy = sin_input(ros_interp_time, freq)
    y *= downfirst
    dy *= downfirst
    plt.plot(ros_interp_time, ros_joint_pos[:, 0], '-', linewidth=2.0)
    plt.plot(ros_interp_time, ros_joint_pos[:, 1], '--', linewidth=2.0)
    # plt.plot(ros_interp_time, y, "g-", linewidth=2.0)
    # plt.plot(ros_interp_time, ros_command_mm[:, 0], "g-", linewidth=2.0)
    # plt.legend(["Tendon 1", "Tendon 2"], fontsize=font_size)
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Displacement (mm)', fontsize=font_size)

    plt.figure(21, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    plt.plot(ros_interp_time, ros_joint_vel[:, 0], "r-", linewidth=2.0)
    plt.plot(ros_interp_time, -ros_joint_vel[:, 1], "b-", linewidth=2.0)
    # plt.plot(ros_interp_time, dy, "g-", linewidth=2.0)
    plt.legend(["Tendon 1 vel", "Negative Tendon 2 vel"], fontsize=font_size)
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Velocity (mm/s)', fontsize=font_size)

    plt.figure(22, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    y, dy = sin_input(ros_interp_time, freq)
    y *= downfirst
    dy *= downfirst
    plt.plot(ros_interp_time, abs(ros_joint_pos[:, 0]-y), "r-", linewidth=2.0)
    plt.plot(ros_interp_time, abs(-ros_joint_pos[:, 1]-y), "b-", linewidth=2.0)
    plt.legend(["Tendon 1-Command", "Negative Tendon 2-Command"], fontsize=font_size)
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Displacement Error(mm)', fontsize=font_size)

    bending_angle_pretension = []
    for xx in range(emData_rot_pretension.shape[0]):
        azimuth = math.radians(emData_rot_pretension[xx, 0])  # Example azimuth angle 1
        elevation = math.radians(emData_rot_pretension[xx, 1])  # Example elevation angle 1
        # bending_radian = math.atan2(math.sin(azimuth) + math.sin(elevation), math.cos(azimuth) + math.cos(elevation))
        x = math.cos(elevation) * math.sin(azimuth)
        y = math.cos(elevation) * math.cos(azimuth)
        z = math.sin(elevation)
        v1 = np.array([x, y, z])
        theta_degrees = cal_rotation(v1)
        bending_angle_pretension.append(theta_degrees)
    bending_angle_pretension = np.array(bending_angle_pretension)
    plt.figure(25, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    # plt.plot(ros_interp_time, bending_angle, linewidth=2.0)
    inx_0s = np.where(ros_interp_time_0 <= 0)[0]
    plt.plot(np.hstack([ros_interp_time_0[inx_0s], ros_interp_time]), np.hstack([bending_angle_pretension[inx_0s], bending_angle]), linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('Tip rotation (degrees)', fontsize=font_size)

    plt.figure(26, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    plt.plot(ros_commandVel_time, ros_command_effort[:, 3]/32767, "-", linewidth=2.0)
    plt.plot(ros_commandVel_time, ros_command_effort[:, 4]/32767, "--", linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('PWM duty cycle command', fontsize=font_size)

    plt.figure(27, figsize=(20, 15))
    plt.tick_params(labelsize=lbs)
    # not aligned with time
    # plt.plot(ros_pwm_time, ros_pwm[:, 3], "-", linewidth=2.0)
    # plt.plot(ros_pwm_time, ros_pwm[:, 4], "--", linewidth=2.0)
    # aligned with time
    plt.plot(ros_interp_time_init, ros_pwm_interp[:, 0], "-", linewidth=2.0)
    plt.plot(ros_interp_time_init, ros_pwm_interp[:, 1], "--", linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize=font_size)
    plt.ylabel('PWM duty cycle', fontsize=font_size)

    # Save Data
    data_to_save = np.column_stack((ros_interp_time, ros_joint_pos, ros_joint_eff, em_pos_interp, em_rot_interp))
    data_to_save = data_to_save[~np.isnan(data_to_save).any(axis=1)]
    data_to_save = np.row_stack((label, data_to_save))
    np.savetxt(os.path.join(directory, item, data_prefix+"{}.txt".format(item.split("_")[-1])), data_to_save, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(directory, data_prefix+"{}.txt".format(item.split("_")[-1])), data_to_save, delimiter=',', fmt='%s')

    # save figure
    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=8)
    plt.plot(ros_interp_time, ros_joint_pos[:, 0], linewidth=1)
    plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
    plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.yticks([0, 1, 2, 3, 4, 5, 6])
    # plt.grid()
    plt.savefig(os.path.join(directory, data_prefix + "{}.png".format(item.split("_")[-1])), dpi=1000)

    # random motion desired input
    trail_path = os.path.join(directory, item, "rosData", "trail_data.csv")
    if os.path.exists(trail_path):
        trail = np.loadtxt(trail_path, delimiter=',', unpack=False, encoding='utf-8')
        max_time = int(trail[0, 2])
        long_pause = [trail[:, 0], trail[:, 1].astype("int")]
        print(max_time, long_pause)
        x_new, y_new, y_de = generate_withDuration(long_pause, max_time, window_size=30)
        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 2))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=8)
        plt.plot(x_new, y_new, linewidth=0.5, label="Desired")
        plt.plot(ros_interp_time, ros_joint_pos[:, 0], linewidth=0.5, label="Actual")
        plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
        plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        # plt.plot(x_new, y_de, label='discrete derivative', linestyle='-', linewidth=1)
        plt.savefig(os.path.join(directory, data_prefix+"{}.png".format(item.split("_")[-1])), dpi=1000)
        # plt.show()

    # random ramp motion desired input
    ramp_trail_path = os.path.join(directory, item, "rosData", "ramp_trail.csv")
    if os.path.exists(ramp_trail_path):
        trail = np.loadtxt(ramp_trail_path, delimiter=',', unpack=False, encoding='utf-8')
        max_time = int(trail[0, 2])
        x, y_true = trail[:, 0], trail[:, 1]
        print(max_time, x, y_true)
        x_new, y_new, y_de = generate_ramp_trail(x, y_true, max_time)
        plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 2))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=8)
        plt.plot(x_new, y_new, linewidth=0.5, label="Desired")
        plt.plot(ros_interp_time, ros_joint_pos[:, 0], linewidth=0.5, label="Actual")
        plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
        plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        # plt.plot(x_new, y_de, label='discrete derivative', linestyle='-', linewidth=1)
        plt.savefig(os.path.join(directory, data_prefix+"{}.png".format(item.split("_")[-1])), dpi=1000)


# Plots
plot_path = os.path.join(directory, 'plots')
os.makedirs(plot_path, exist_ok=True)
legend_list = ["Trial{}".format(i) for i in range(1, 10)]
legend_list2 = []
for i in range(1, 10):
    legend_list2.append("T{}_td1".format(i))
    legend_list2.append("T{}_td2".format(i))
# legend_list = ["0.{} Hz".format(i) for i in range(1, 10)]
fig_name = ["disp1_tipDisp", 'disp1_rot', 'disp1_rotAE', 'curr1_tipDisp', 'curr1_rot', 'curr1_tendonDisp1', 'time_curr1', 'time_command1', 'Angle_tipXY',
            "disp2_tipDisp", 'disp2_rot', 'disp2_rotAE', 'curr2_tipDisp', 'curr2_rot', 'curr2_tendonDisp2', 'time_curr2', 'time_command2', 'Angle_tipXY',
            'tip_position', 'time_tendonDisplacement', 'time_tendonVelocity', 'time_tendonDisplacement_error',
            'disp1_rotE', 'disp2_rotE', "time_rotAE", "time_pwmCommand", "time_dutyCycle"]
for i in range(27):
    plt.figure(i+1)
    if (i < 19 or i >= 22) and i!=8 and i!=17:
        plt.legend(legend_list, fontsize=font_size)
    if i==8 or i==17:
        plt.legend(["Measured Distance", "Constant Curvature"], fontsize=font_size, frameon=False)
    if i==19:
        plt.legend(legend_list2, fontsize=font_size)
    if i==25 or i==26:
        plt.legend(legend_list2, fontsize=font_size)
    plt.savefig(os.path.join(plot_path, '{}.png'.format(fig_name[i])))


# for i in [4, 7, 8, 9, 12, 15, 16, 17]:
for i in range(23):
    plt.close(i)
# plt.show()