import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

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

def generate_random(t):
    max_time = 30
    y_true = [0, 1, 4, 5, 5.1, 5.1, 5.1, 5, 1, 5.1, 5, 3, 2, 4, 6, 5.5, 3, 3.1, 3, 2, 3, 5, 4, 1, 3, 4, 3.8, 3.5, 2, 1, 0]
    x = np.linspace(0, max_time, len(y_true))
    x_new = np.linspace(0, max_time, 1000)
    y_new = np.interp(x_new, x, y_true)
    window_size = 50
    smooth_window = np.ones(window_size) / window_size
    y_new = np.convolve(y_new, smooth_window, mode='same')
    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    c = 0
    for i in range(len(x_new)-1):
        if (t >= x_new[i]) and (t < x_new[i+1]):
            c = y_de[i]
    return c



if __name__ == '__main__':
    time = np.linspace(0, 70, 1000)
    max_time = 30
    y_true = [0, 0.1, 0.2, 0.4, 1, 4, 5, 5.1, 5.1, 5.1, 5, 1, 5.1, 5, 3, 2, 3, 6, 5.5, 3, 3.1, 3, 2, 3, 5, 4, 1, 3, 4, 3.8, 3.5, 2, 1, 0.5, 0.2, 0]  # Test random1
    y_true = [0, 0.2, 0.4, 1, 3, 4, 5.1, 5, 3, 2, 1, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2.5, 2, 1, 2, 2.8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 3, 4, 3.8, 3.5, 2, 1, 0.5, 0.2, 0]  # Test random1
    x = np.linspace(0, max_time, len(y_true))
    x_new = np.linspace(0, max_time, 1000)
    y_new = np.interp(x_new, x, y_true)
    window_size = 50
    smooth_window = np.ones(window_size) / window_size
    y_new = np.convolve(y_new, smooth_window, mode='same')
    y_de = (y_new[1:]-y_new[:-1])/(x_new[1:]-x_new[:-1])

    # Fit a 10th-order polynomial
    coefficients = np.polyfit(x_new, y_new, 100)
    # print(len(coefficients))
    derivative_coefficients = np.polyder(coefficients)
    # print(derivative_coefficients)
    poly_fit = np.poly1d(coefficients)

    # Evaluate the polynomial at the original x values
    y_fit = poly_fit(x_new)
    t = np.linspace(0, 30, 2000)
    cmd_list = []
    for j in t:
        cmd_list.append(generate_random(j))
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_new, y_new, label='random signal', linestyle='--', linewidth=2)
    plt.plot(x_new[:-1], y_de, label='discrete derivative', linestyle='-', linewidth=2)
    # plt.plot(t, cmd_list, label='discrete derivative', linestyle='-', linewidth=2)
    # plt.plot(x_new, y_fit, label='Fitted Polynomial', linewidth=2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()