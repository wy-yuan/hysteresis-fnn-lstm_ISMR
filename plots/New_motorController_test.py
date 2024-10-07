import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import pandas as pd

if __name__ == '__main__':
    file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\sinusoid_test_trial11.csv"
    file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\sinusoid_test_pololu_trial4_A20.csv"
    file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\test_current\current_test_6A_norobot_trial3.csv"
    file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\test_current\current_test_6A_robot_trial3.csv"
    file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\test_current\current_test_127PWM_norobot_trial1.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\test_current\current_test_16PWM_norobot_trial1.csv"
    file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\data.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P09I003D005.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial4_P07I002D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P1I002D0.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P1_5I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P1_5I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial3_P1_5I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial4_P1_5I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P3000I30D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P10I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P3000I150D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P100I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial3_P100I0_1D000_robot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P0_7I0_02D000_defaccdec200.csv"

    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P0_7I0_02D000_withrobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P10I0_1D000_withrobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial3_P100I0_1D000_withrobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial3_P3000I30D300_withrobot.csv"


    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P27_1I2_4D000_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial3_P27_1I2_4D000_speed100_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P27_1I2_4D000_withrobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P2_71I0_24D000_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P2_71I0_24D000_withrobot.csv"

    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P0_271I0_024D000_speed15_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P0_71I0_02D000_speed15_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P2_71I0_24D000_speed15_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P2_71I0_24D000_speed3_norobot_pololu.csv"

    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial2_P0_271I0_024D000_speed3_norobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P0_271I0_024D000_speed3_withrobot.csv"
    # file = r"D:\yuan\Dropbox (BCH)\robot debugging experiments\motor tuning\motor_tuning_trial1_P0_271I0_024D000_speed6_withrobot.csv"


    df = pd.read_csv(file)
    # print(df)

    time = np.array(df["time"])
    ros_joint_pos = np.array(df["position"])
    ros_joint_vel = np.array(df["velocity"])
    ros_joint_cur = np.array(df["current"])

    # actual_vel = np.ones_like(time) * 2
    time = time - time[0]
    flag = np.ones_like(time)
    flag[time > 10] = 0
    actual_vel = 3*2*np.pi*0.1*np.cos(2*np.pi*0.1*time-np.pi/2)*7420/80.42 * flag
    # actual_vel = 3*2*np.pi*0.1*np.cos(2*np.pi*0.1*time-np.pi/2)*979/80.42 * flag
    # actual_vel = 20*2*np.pi*0.1*np.cos(2*np.pi*0.1*time-np.pi/2)*832/80.42 * flag
    actual_pos = np.cumsum(actual_vel * (np.hstack([time[1:], time[-1]]) - time))
    actual_pos_from_command = np.cumsum(ros_joint_vel * (np.hstack([time[1:], time[-1]]) - time))


    font_size = 10
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax1 = plt.subplots(figsize=(88.9 / 25.4 * 4, 88.9 / 25.4*2))
    ax1.tick_params(labelsize=font_size, pad=0.01, length=2)
    ax1.plot(time, ros_joint_vel)
    ax1.plot(time, actual_vel)
    # ax1.plot(ros_joint_vel)
    ax1.set_xlabel('Time (s)', fontsize=font_size)
    ax1.set_ylabel('Velocity (count/s)', fontsize=font_size)
    # ax1.grid()

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=font_size, pad=0.01, length=2)
    ax2.plot(time, actual_pos, 'r')
    # ax2.plot(time, (ros_joint_pos[:, motor] - ros_joint_pos[0, motor]), 'g')
    # ax2.plot(time, actual_pos-(ros_joint_pos[:, motor] - ros_joint_pos[0, motor]), 'y')
    ax2.plot(time, (ros_joint_pos - ros_joint_pos[0]), 'g')
    ax2.plot(time, actual_pos - (ros_joint_pos - ros_joint_pos[0]), 'y')
    ax2.plot(time, actual_pos_from_command, 'purple')
    # ax2.plot(time, ros_joint_cur, 'r')
    # ax2.plot(ros_joint_pos, 'r')
    ax2.tick_params('y', colors='g')
    ax2.set_ylabel('Position (count)', fontsize=font_size)

    # plt.savefig(r"D:\yuan\Dropbox (BCH)\robot debugging experiments\plots\{}.png".format(file.split(".")[0]),dpi=600)
    plt.savefig(r"{}.png".format(file.split(".")[0]), dpi=600)
    plt.show()