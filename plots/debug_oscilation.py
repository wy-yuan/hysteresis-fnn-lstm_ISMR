import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import pandas as pd

if __name__ == '__main__':

    directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus01_noOscilation_01"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus01_noOscilation_02"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus01_noOscilation_03"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus01_oscilateAfterFirstCycle"

    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus05_oscilateAfterFirstCycle"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus05_oscilationAfterFirstCycle_01"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus05_oscilationAfterFirstCycle_02"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first01_cus05_oscilationAfterFirstCycle_03"

    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first05_cus01_oscilateDuringFirstCycle_01"
    # directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first05_cus01_oscilateDuringFirstCycle_02"
    directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\rosbag2_0802_exp6_first05_cus01_oscilateImmediately"

    directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\0808\rosbag2_0808_m7_m8_I0_3_01HZ_withrobot_pretension120_trial4"
    directory = r"D:\yuan\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\0808\rosbag2_0808_m7_I0_3_01HZ_withrobot_Pretension120_trial1"
    # directory = r"C:\Users\wangyuan\Documents\CV\BCH Dropbox\Wang Yuan\robot debugging experiments\debug_oscilation\0808\rosbag2_0808_m7_m8_I0_3_01HZ_withrobot_pretension120_trial4"

    joint_data_feedback_file = "joint_data_feedback.csv"
    commanded_velocity_file = "joint_data_vel_command.csv"
    # commanded_velocity_file = "commanded_velocity.csv"
    # cvc_motion_file = "commanded_velocity_custom_motion.csv"
    EM_trigger_file = "EM_trigger_running_time.csv"
    motor = 7
    rosData = {}

    for file_name in [joint_data_feedback_file, commanded_velocity_file, EM_trigger_file]:
    # for file_name in [joint_data_feedback_file, commanded_velocity_file, cvc_motion_file, EM_trigger_file]:
        var_name = file_name.split(".")
        sheet = pd.read_csv(os.path.join(directory, file_name))
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

    # commanded_velocity data
    commanded_velocity_name = "joint_data_vel_command"  # 'commanded_velocity'
    cv_time = np.array(rosData[commanded_velocity_name]['time'])
    # cv_time = cv_time - cv_time[0]
    cv_ros_joint_vel = np.array(rosData[commanded_velocity_name]['velocity'])

    # commanded_velocity_custom_motion
    # cvc_time = np.array(rosData['commanded_velocity_custom_motion']['time'])
    # cvc_time = cvc_time - cvc_time[0]
    # cvc_ros_joint_vel = np.array(rosData['commanded_velocity_custom_motion']['velocity'])

    # EM time
    ros_interp_time = rosData['EM_trigger_running_time']['time']

    time = np.array(rosData['joint_data_feedback']['time'])
    # time = time - time[0]
    ros_joint_pos = np.array(rosData['joint_data_feedback']['position'])*7420/2/np.pi/12.8
    ros_joint_vel = np.array(rosData['joint_data_feedback']['velocity'])*7420/2/np.pi/12.8
    ros_joint_eff = np.array(rosData['joint_data_feedback']['effort'])

    # align time with EM trigger
    ros_joint_pos = np.interp(ros_interp_time, time, ros_joint_pos[:, motor])
    ros_joint_vel = np.interp(ros_interp_time, time, ros_joint_vel[:, motor])
    cv_ros_joint_vel = np.interp(ros_interp_time, cv_time, cv_ros_joint_vel[:, motor])

    if np.any(np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0):
        ros_joint_pos = ros_joint_pos[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
        ros_joint_vel = ros_joint_vel[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
        cv_ros_joint_vel = cv_ros_joint_vel[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
        ros_interp_time = np.array(ros_interp_time)
        EM_time = ros_interp_time[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
        EM_time = EM_time - EM_time[0]
        time = EM_time
    else:
        time = ros_interp_time

    font_size = 10
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax1 = plt.subplots(figsize=(88.9 / 25.4*4, 88.9 / 25.4*2))
    ax1.tick_params(labelsize=font_size, pad=0.01, length=2)
    ax1.plot(time, -ros_joint_vel)  # plot actual velocity
    ax1.plot(time, -cv_ros_joint_vel, 'k')   # plot command velocity
    cv_ros_joint_pos = np.cumsum(cv_ros_joint_vel * (np.hstack([time[1:], time[-1]]) - time))
    ax1.plot(time, -(ros_joint_pos - ros_joint_pos[0]))  # plot actual pos
    ax1.plot(time, -cv_ros_joint_pos, 'k')  # plot command pos

    ax1.set_xlabel('Time (s)', fontsize=font_size)
    # ax1.set_ylabel('Velocity (count/s)', fontsize=font_size)
    ax1.set_ylabel('Position (count)', fontsize=font_size)
    # ax1.set_xlim([-2, 30])

    plt.savefig(r"{}.png".format(directory.split("rosbag2_")[-1]), dpi=600)
    plt.show()