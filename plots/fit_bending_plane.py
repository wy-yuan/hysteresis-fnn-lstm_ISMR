import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math

def cal_rotation(v1, normal_vector=[-0.76435347,-0.13500009,0.61676366]):
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


def fit_plane():
    data_dir = r"D:\yuan\BCH Dropbox\Wang Yuan\LearningDatasets\InnerSheath2024\0907\for_bending_plane_calculation"
    bending_vector = []
    for file_name in os.listdir(data_dir):
        data_path = os.path.join(data_dir, file_name)
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        em_rot_interp = data[:, 8:11]
        for xx in range(em_rot_interp.shape[0]):
            azimuth = math.radians(em_rot_interp[xx, 0])  # Example azimuth angle 1
            elevation = math.radians(em_rot_interp[xx, 1])  # Example elevation angle 1
            # bending_radian = math.atan2(math.sin(azimuth) + math.sin(elevation), math.cos(azimuth) + math.cos(elevation))
            x = math.cos(elevation) * math.sin(azimuth)
            y = math.cos(elevation) * math.cos(azimuth)
            z = math.sin(elevation)
            bending_vector.append([x, y, z])
    bending_vector = np.array(bending_vector)
    print(bending_vector.shape)
    A = np.hstack([bending_vector, np.ones((bending_vector.shape[0], 1))])
    print(A.shape)
    # Perform SVD to find the plane coefficients
    U, S, Vt = np.linalg.svd(A)
    plane_coefficients = Vt[-1, :]  # Last row of V^T gives the smallest singular value
    # Normal vector to the plane
    normal_vector = plane_coefficients[:3]  # The normal vector is the first three components
    d = plane_coefficients[3]
    print(normal_vector)

    converted_angle = []
    data_path = r"D:\yuan\BCH Dropbox\Wang Yuan\LearningDatasets\InnerSheath2024\0904\antagonisticA2_5\pre_1.txt"
    data_path = r"D:\yuan\BCH Dropbox\Wang Yuan\LearningDatasets\InnerSheath2024\0907\for_bending_plane_calculation\pre_1.txt"
    data_path = r"D:\yuan\BCH Dropbox\Wang Yuan\LearningDatasets\InnerSheath2024\0907\PWM3_9-3_reducedPWM0_9\pre_1.txt"
    data_path = r"D:\yuan\BCH Dropbox\Wang Yuan\LearningDatasets\InnerSheath2024\0907\PWMlist\pre_1.txt"
    data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    em_rot_interp = data[:, 8:11]
    for xx in range(em_rot_interp.shape[0]):
        azimuth = math.radians(em_rot_interp[xx, 0])  # Example azimuth angle 1
        elevation = math.radians(em_rot_interp[xx, 1])  # Example elevation angle 1
        # bending_radian = math.atan2(math.sin(azimuth) + math.sin(elevation), math.cos(azimuth) + math.cos(elevation))
        x = math.cos(elevation) * math.sin(azimuth)
        y = math.cos(elevation) * math.cos(azimuth)
        z = math.sin(elevation)
        v1 = np.array([x, y, z])

        theta_degrees = cal_rotation(v1)

        converted_angle.append(theta_degrees)
    plt.plot(data[:, 1], converted_angle)
    plt.show()
    plt.plot(data[:, 0], converted_angle)
    plt.show()

def cal_pwm_with_friction(direction=2):
    t = np.linspace(0, 70, 1000*70)
    freq = 0.1
    pretension1 = 2.3-0.5
    pretension2 = 2
    friction1 = 2
    friction2 = 1.5
    if direction == 1:
        fd = 1*(np.sin(2 * np.pi * freq * t - np.pi / 2)+1)
        fd_der = 1*2 * freq * np.cos(2 * np.pi * freq * t - np.pi / 2)
        pwm1 = np.zeros_like(fd)
        pwm2 = np.zeros_like(fd)
        fd1 = fd + pretension1
        fd2 = np.zeros_like(fd)-pretension2
        # fd2 = -(np.sin(2 * np.pi * f * t - np.pi / 2) + 1) - pretension1
        for i in range(fd1.shape[0]):
            if fd_der[i] > 0:
                pwm1[i] = fd1[i] + friction1
                pwm2[i] = fd2[i] + friction2
            else:
                pwm1[i] = fd1[i] - friction1
                pwm2[i] = fd2[i] - friction2
        pwm2[0] = fd2[0] - friction2
    else:
        # fd = 1 * 2 * np.sin(2*np.pi*freq*t)
        # fd_der = 1 * 4 * freq * np.cos(2 * np.pi * freq * t)
        tau = (-freq / 4) * np.log(2 / 14)
        fd = 2 * np.exp(-tau * t) * np.sin(2 * np.pi * freq * t)
        fd_der = 2 * np.exp(-tau * t) * (-tau * np.sin(2 * np.pi * freq * t) + 2 * np.pi * freq * np.cos(2 * np.pi * freq * t))
        pwm1 = np.zeros_like(fd)
        pwm2 = np.zeros_like(fd)
        fd1 = fd + pretension1
        fd2 = fd - pretension2
        for i in range(fd1.shape[0]):
            if fd[i]>=0 and fd_der[i] >= 0:
                pwm1[i] = fd1[i] + friction1
                pwm2[i] = -pretension2 + friction2
            elif fd[i]>=0 and fd_der[i] < 0:
                pwm1[i] = fd1[i] - friction1
                pwm2[i] = -pretension2 - friction2
            elif fd[i]<0 and fd_der[i] < 0:
                pwm1[i] = pretension1 - friction1
                pwm2[i] = fd2[i] - friction2
            elif fd[i]<0 and fd_der[i] >= 0:
                pwm1[i] = pretension1 + friction1
                pwm2[i] = fd2[i] + friction2
        pwm2[0] = fd2[0] - friction2

    # add pretension period to the smooth results
    pwm1 = np.concatenate([np.ones(1000)*(pretension1+friction1), pwm1])
    pwm2 = np.concatenate([np.ones(1000)*(-pretension2-friction2), pwm2])

    window_size = 500
    pwm1 = np.convolve(pwm1, np.ones(window_size)/window_size, mode='valid')
    pwm2 = np.convolve(pwm2, np.ones(window_size) / window_size, mode='valid')
    # pwm2 = fd1 - f1
    plt.plot(pwm1, label="tendon1")
    plt.plot(pwm2, label="tendon2")
    plt.xlabel("time")
    plt.ylabel("PWM (%)")
    plt.legend()
    save_path = r"D:\yuan\BCH Dropbox\Wang Yuan\bch\projects\hysteresis-fnn-lstm_ISMR\figures\Antagonistic_tendons"
    plt.savefig(os.path.join(save_path, "{}.png".format("cal_pwm_two_direction")), dpi=1000)
    plt.show()

if __name__ == '__main__':
    # fit_plane()
    cal_pwm_with_friction()