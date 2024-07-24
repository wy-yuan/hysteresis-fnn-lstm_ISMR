import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import pandas as pd

# Example DataFrame
data = {
    'header': [1, 2, 3, 4],
    'position': ['A', 'B', 'C', 'D'],
    'velocity': [1, 2, 3, 4]
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('output.csv', index=False)
df = pd.read_csv('output.csv')

if __name__ == '__main__':
    directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_11_vel_resolution_exp2_withvelocity"
    directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_10_norobot_onecycle_trial1"
    directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_1"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_withrobot_1"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_staircase4_1"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_staircase3_1"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_staircase2_1"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_staircase1_1"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_withrobot_pretension_2"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_check_calc_velocity_staircase5_mult50"
    # directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_const_vel_1"
    directory = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\rosbag_6_12_const_vel_10"

    directory = r"D:\yuan\Dropbox (BCH)\LearningDatasets\Data_innerTendonSlack\2024-02-22\0_baseline\rosbag2_2024-02-22_18-07-43_0BL_0.1Hz\rosData"


    joint_data_feedback_file = "joint_data_feedback.csv"
    commanded_velocity_file = "joint_data_vel_command.csv"
    # commanded_velocity_file = "commanded_velocity.csv"
    # cvc_motion_file = "commanded_velocity_custom_motion.csv"
    EM_trigger_file = "EM_trigger_running_time.csv"
    motor = 6
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
    ros_joint_pos = ros_joint_pos[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    ros_joint_vel = ros_joint_vel[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    cv_ros_joint_vel = cv_ros_joint_vel[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    ros_interp_time = np.array(ros_interp_time)
    EM_time = ros_interp_time[np.array(rosData['EM_trigger_running_time']['effort'])[:, 0] > 0]
    EM_time = EM_time - EM_time[0]
    time = EM_time
    # actual_vel = (np.floor(time / 5) + 1)*50
    # actual_vel = (np.floor(time /3) + 1)
    # actual_vel = np.ones_like(time)*10
    flag = np.ones_like(time)
    flag[time>10] = 0
    actual_vel = 3*2*np.pi*0.1*np.cos(2*np.pi*0.1*time-np.pi/2)*7420/80.42 * flag
    actual_pos = np.cumsum(actual_vel*(np.hstack([time[1:], time[-1]])-time))
    actual_pos_from_command = np.cumsum(ros_joint_vel*(np.hstack([time[1:], time[-1]])-time))

    font_size = 10
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax1 = plt.subplots(figsize=(88.9 / 25.4*4, 88.9 / 25.4*2))
    ax1.tick_params(labelsize=font_size, pad=0.01, length=2)
    ax1.plot(time, ros_joint_vel)
    print("average velocity:", np.mean(ros_joint_vel))
    ax1.plot(time, actual_vel)
    ax1.set_xlabel('Time (s)', fontsize=font_size)
    ax1.set_ylabel('Velocity (count/s)', fontsize=font_size)

    ax1.plot(time, cv_ros_joint_vel, 'k')
    ax1.set_xlim([-0.1, 10.1])

    # ax1.plot(cv_time, cv_ros_joint_vel[:, motor], 'k')
    # ax1.plot(cvc_time, cvc_ros_joint_vel[:, motor], 'purple')

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=font_size, pad=0.01, length=2)
    ax2.plot(time, actual_pos, 'r')
    # ax2.plot(time, (ros_joint_pos[:, motor] - ros_joint_pos[0, motor]), 'g')
    # ax2.plot(time, actual_pos-(ros_joint_pos[:, motor] - ros_joint_pos[0, motor]), 'y')
    ax2.plot(time, (ros_joint_pos - ros_joint_pos[0]), 'g')
    ax2.plot(time, actual_pos - (ros_joint_pos - ros_joint_pos[0]), 'y')
    ax2.plot(time, actual_pos_from_command, 'purple')
    ax2.tick_params('y', colors='g')
    ax2.set_ylabel('Position (count)', fontsize=font_size)
    ax2.set_xlim([-0.1, 10.1])
    plt.savefig(r"{}.png".format(directory.split("rosbag_")[-1]), dpi=600)
    print(r"{}.png".format(directory.split("rosbag_")[-1]))
    plt.show()

