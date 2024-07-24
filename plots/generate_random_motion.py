import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import sys
sys.path.append("..")
from train_TC import LSTMNet
from train_TC import FFNet
import torch

def test_LSTM(pt_path, input_data, seg=50, sr=25, input_dim=1, forward=True):
    device = "cuda"
    model = LSTMNet(inp_dim=input_dim, num_layers=2)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    if forward:
        input_data = np.hstack([-np.ones(seg), input_data/6])
    else:
        input_data = np.hstack([-np.ones(seg), input_data / 90])
    joints = input_data[:, np.newaxis].astype("float32")

    out = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(len(input_data)):
        joint = joints[i:i + 1, 0:input_dim]
        input_ = joint
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)
    if forward:
        output = out[seg:, 0]*90
    else:
        output = out[seg:, 0]*6
    return output

def test_FNN(pt_path, input_data, seg=50, sr=25, forward=True, input_dim=1):
    device = "cuda"
    model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg)
    sr = sr
    seg = seg
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.cuda()
    model.eval()

    if forward:
        input_data = np.hstack([-np.ones(seg), input_data / 6])
    else:
        input_data = np.hstack([-np.ones(seg), input_data / 90])
    joints = input_data[:, np.newaxis].astype("float32")

    out = []
    for i in range(len(input_data) - seg):
        joint = joints[i + 1:i + seg + 1, 0:input_dim]
        input_ = joint
        output = model(torch.tensor([input_]).to(device))
        predict_pos = output.detach().cpu().numpy()[0]
        out.append(predict_pos[0])
    out = np.array(out)
    if forward:
        output = out[:, 0]*90
    else:
        output = out[:, 0] * 6
    return output

def sin_input_signal(x, f, type="non_decay"):
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

    if type == "0bl_inc":
        y = A * np.exp(-tau * (7/f-x)) * (np.sin(2 * np.pi * f * (7/f-x) - np.pi / 2) + 1)
        dy = A * tau * np.exp(-tau * (7/f-x)) * (np.sin(2 * np.pi * f * (7/f-x) - np.pi / 2) + 1) - A * np.exp(
            -tau * (7/f-x)) * 2 * np.pi * f * np.cos(2 * np.pi * f * (7/f-x) - np.pi / 2)

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

    return y, dy

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

def generate(long_pause):
    y_true = long_pause
    x = np.linspace(0, max_time, len(y_true))
    x_new = np.linspace(0, max_time, 1000)
    x_new_ = np.linspace(0, max_time, 1049)
    y_new = np.interp(x_new, x, y_true)
    window_size = 50
    smooth_window = np.ones(window_size) / window_size
    y_new = np.convolve(y_new, smooth_window, mode='same')
    # y_new = y_new - y_new[0]  #  make tendon displacement start from 0
    # y_new = (y_new - y_new[0])*12+20  #  make tendon displacement start from 0
    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    return x_new, y_new, y_de

def generate_withDuration(long_pause, max_time, window_size=30):
    sig = long_pause[0]
    repeats = long_pause[1]
    y_true = np.array([])
    for s,r in zip(sig, repeats):
        y_true = np.hstack([y_true, np.ones(r)*s])

    # window_size = 1
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

def get_tendonDisp(t, x_new, y_de):
    c = 0
    for i in range(len(x_new) - 1):
        if (t >= x_new[i]) and (t < x_new[i + 1]):
            c = y_de[i]
            break
    return c

def generate_random_trail():
    duration = np.random.randint(15, 60, 1)[0]
    pos_number = np.random.randint(3, 15, 1)[0]
    position = np.random.rand(pos_number)*6
    pause = np.random.randint(1, 10, pos_number)
    return [np.hstack([0, position]), np.hstack([2, pause])], duration


def generate_ramp_trail(max_time):
    pos_number = np.random.randint(5, 10, 1)[0]
    position = np.random.rand(pos_number) * 6
    interval = np.random.rand(pos_number)*3+1
    interval_time = interval/sum(interval)*max_time

    x = np.hstack([0, np.around(np.cumsum(interval_time), decimals=2)])
    y_true = np.hstack([0, position])
    x_new = np.arange(0, max_time, 0.01)
    y_new = np.interp(x_new, x, y_true)

    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    y_de = np.hstack([np.zeros(1), y_de])

    return [x_new, y_new, y_de]

    # plt.plot(x_new, y_new, linestyle='-', linewidth=1)
    # plt.plot(x_new, y_de, linestyle='-', linewidth=1)
    # plt.show()

def generate_staircase(slope, stair, slope_rd=False, stair_rd=False, duration=[1,10]):
    max_time = 0
    if slope_rd:
        slopes = np.random.rand(11)*(6-0.5) + 0.5   #~[0.6*np.pi, 3*np.pi]
        # slopes = np.hstack([0.5, slopes])
    else:
        slopes = np.ones(11)*slope

    if stair_rd:
        stairs = [np.random.rand(1)*6]
        while len(stairs) < 11:
            stairs_i = np.random.rand(1)*6
            if 0.5 < abs(stairs_i-stairs[-1]) < 3:
                stairs.append(stairs_i)
    else:
        stairs = stair
    stairs_durations = np.random.rand(11) * (duration[1]-duration[0]) + duration[0]  # stationary period duration ~[10, 60] second
    x = [0]
    y = [0]
    stair_pre = 0
    for i in range(len(stairs)):
        slope = slopes[i]
        slope_duration = abs(stairs[i]-stair_pre)/slope
        stair_pre = stairs[i]
        max_time = max_time + slope_duration
        x = np.hstack([x, max_time])
        y = np.hstack([y, stairs[i]])
        stair_duration = stairs_durations[i]
        max_time = max_time + stair_duration
        x = np.hstack([x, max_time])
        y = np.hstack([y, stairs[i]])

    x_new = np.arange(0, max_time, 0.01)
    y_new = np.interp(x_new, x, y)

    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    y_de = np.hstack([np.zeros(1), y_de])

    return x_new, y_new, y_de, max_time

def generate_rd(min_val, max_val, count, threshold):
    numbers = []
    while len(numbers) < count:
        num = np.random.rand() * (max_val-min_val) + min_val
        if all(abs(num - other) > threshold for other in numbers):
            numbers.append(num)
    return numbers
# stair_times = np.sort(np.array(generate_rd(0, max_time*100, 10, max_time*5)).astype(int))
def generate_sinus_stationary(sinusType, f, duration=[1, 10]):
    max_time = 7/f
    t = np.arange(0, max_time, 0.01)
    sy, dsy = sin_input_signal(t, f, type=sinusType)

    y_new = sy
    stair_durations = np.array((np.random.rand(10) * (duration[1] - duration[0]) + duration[0])*100).astype(int)
    stair_times = np.sort(np.array(np.random.rand(10)*max_time*100).astype(int))

    for i in range(10):
        stair_time = stair_times[i]
        added = sum(stair_durations[:i])
        index = stair_time+added
        y_new = np.hstack([y_new[:index], np.ones(stair_durations[i])*sy[stair_time], y_new[index:]])
    x_new = np.linspace(0, len(y_new)/100, len(y_new))
    y_de = (y_new[1:] - y_new[:-1]) / (x_new[1:] - x_new[:-1])
    y_de = np.hstack([np.zeros(1), y_de])
    return x_new, y_new, y_de

if __name__ == '__main__':
    # increasing sinusoidal signal
    t = np.linspace(0, 14, 1000)
    y, dy = sin_input_signal(t, 0.5, type="0bl_inc")
    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(t, y, linestyle='-', linewidth=1)
    plt.plot(t, dy, linestyle='-', linewidth=1)
    # plt.show()

    # staircase with different slopes, each trail with one slope
    slope, stair = [], []
    x_new, y_new, y_de, max_time = generate_staircase(slope, stair, slope_rd=True, stair_rd=True, duration=[1, 5])
    # slope, stair = 0.1, [1,2,3,4,5,6,5,4,3,2,1]
    # x_new, y_new, y_de, max_time = generate_staircase(slope, stair, slope_rd=False, stair_rd=False, duration=[1,10])

    # slope, stair = [], [1,2,3,4,5,6,5,4,3,2,1]
    # slope, stair = [], [3,2,1,0,1,2,3,4,5,6,5]
    # slope, stair = [], [6,5,4,3,2,1,2,3,4,3,2]
    # x_new, y_new, y_de, max_time = generate_staircase(slope, stair, slope_rd=True, stair_rd=False, duration=[10,60])

    # x_new, y_new, y_de = generate_sinus_stationary("0bl", 0.2, [1, 10])
    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(x_new, y_new, linestyle='-', linewidth=1)
    plt.plot(x_new, y_de, label='discrete derivative', linestyle='-', linewidth=2)
    plt.show()
    # staircase with mixed slopes for testing


    max_time = 70
    y_true = [0, 0.1, 0.2, 0.4, 1, 4, 5, 5.1, 5.1, 5.1, 5, 1, 5.1, 5, 3, 2, 3, 6, 5.5, 3, 3.1, 3, 2, 3, 5, 4, 1, 3, 4, 3.8, 3.5, 2, 1, 0.5, 0.2, 0]  # Test random1
    y_true = [0, 0.2, 0.4, 1, 3, 4, 5.1, 5, 3, 2, 1, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2.5, 2, 1, 2, 2.8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 3, 4, 3.8, 3.5, 2, 1, 0.5, 0.2, 0]  # Test random1
    # y_true = [0, 1, 4, 5, 5.1, 5.1, 5.1, 5, 1, 5.1, 5, 3, 2, 4, 6, 5.5, 3, 3.1, 3, 2, 3, 5, 4, 1, 3, 4, 3.8, 3.5, 2, 1, 0]
    y_true = [0, 1, 2, 3, 4, 5.4, 5.4,5.4,5.4,5.4,5.4,5.4,5.4,4,3,2,1,0, 1,2,3,4,4,4,4,4,3,2,1,0.3,0.3,0.3,0.3, 0.3, 0.3, 0.3, 1,2.2,2.2,2.2,2.2,2.2,2.2,2.2, 1.6,1,0.5, 0.3,0.3,0.3,1]
    random_pause1 = [0, 1, 2, 3, 4, 5, 5,5,5,5,5,5,5,4,3,2,1,0.3, 0.3,0.3, 1,2,3,3.6,3.6,3.6,3.6,3.6,3.6,3.6,3,2,1,0.3,0.3,0.3,0.3, 0.3, 0.3, 0.3, 1,2,2,2,2,2,2,2, 1.6,1,0.5, 0.3,0.3,0.3,1]
    random_pause2 = [0, 1, 3, 5, 5.4,5.4,5.4,5.4,5.4,5.4,5.4,5, 4,3,2,1,1, 1,2,3,4,4.5,4.5,4.5,4.5,4.5,4.5,4.5,3,2.5,2,1.5,1.5,2,3.5,3.5,3.5,3.5,3.5,3.5,3,2.5,2,1.5,1]
    random_pause3 = [0, 1, 2, 3, 4, 5, 5.4,5.4,5.4,5.4,5.4,5.4,5.4,5, 4,3,2,1,1, 1, 1,2,3,4,5.4, 5.4, 5.4,5.4,5.4,5.4,5,4,3,3,2.5,3,4,4.5,5,5.4,5.4,5.4,5.4,5.4,5,5,4.5,4,3,2]
    random_pause4 = [0, 1, 2, 3,3,3,3,3,3,3,3,3,3,3,3, 4, 5, 5.4,5.4,5.4,5.4,5.4,5.4,5.4,5, 4,3,2,1,1, 1, 1,2,3,4,5.4, 5.4, 5.4,5.4,5.4,5.4,5,4,3,3,2.5,3,4,4.5,5,5.4,5.4,5.4,5.4,5.4,5,5,4.5,4,3,2]
    random_pause5 = [0, 1, 2, 3, 4, 5, 5.4,5.4,5.4,5.4,5.4,5.4,5.4, 5, 4,3,2,2,2,2,2,2,2,2,2,2,2,2,3,4,5.4, 5.4, 5.4,5.4,5.4,5.4,5,4,3,3,2.5,3,4,4.5,5]

    long_pause1 = [0,0,0,1,2,3,4,5,6,6,6,6,6,6,5,4,3,2,1,0,0,0,0,0,0,1,2,3,4,4,4,4,4,4,3,2,1,0,0,0,0,0,0,1,2,2,2,2,2,2,1,0,0,0,0,0,0]
    long_pause2 = [0,0,0,1,2,3,4,5,5.8,5.8,5.8,5.8,5.8,5.8,5,4,3,2,1,1,1,1,1,1,2,3,4.7,4.7,4.7,4.7,4.7,4.7,3,2,2,2,2,2,2,3.5,3.5,3.5,3.5,3.5,3.5,3,2.5,2.5,2.5,2.5,2.5]
    long_pause3 = [0,0,0,1,2,3,4,5,6,6,6,6,6,6,5,4,3,2,1.5,1.5,1.5,1.5,1.5,1.5,2,3,4,5,6,6,6,6,6,6,5,4,3,3,3,3,3,3,4,5,6,6,6,6,6,6,5,4,4,4,4,4,4]
    long_pause4 = [0,0,0,0,0,1,2,2,2,2,2,2,2,2,2,3,4,5,6,5,4,3,2,2,2,2,2,2,2,2,2,1,0,0,0,0,1,2,3,3,3,3,3,3,3,3,3,4,5,6,5,4,3,3,3,3,3,3,3,3,3,3,2,1,0,0,0,0,1,2,3,4,4,4,4,4,4,4,4,4,5,6,5,4,4,4,4,4,4,4,4,4,3,1,0,0,0,0,0]
    long_pause5 = [[0, 1, 2,  3, 4,  5.5, 6, 5.5, 4, 3,  2, 1,  0, 1, 3,  4, 3, 2.5, 2, 1, 0, 1, 2, 1.5, 1, 0],
                   [10, 1, 1, 20, 1, 1,  25, 1,   1, 1, 15, 1, 30, 1, 1, 45, 30, 1, 1,  1,  15, 1, 18, 1, 1, 10]]
    long_pause6 = [[0, 1, 2, 3, 4, 5, 5.8, 5, 4, 3.5, 3, 2, 1, 2, 2.5, 3, 4.7, 3, 2, 2.5, 3.5, 3, 2.5],
                   [5, 1, 1, 1, 1, 1, 20,  2, 1,  10, 1, 1, 25, 1, 7, 1, 15,  1, 10, 1,  10, 1, 5]]
    long_pause7 = [[0,1,2,3,4,5,   6,5,4,3,2,    1.5,2,3,4,5,  6,5,4,  3,4,5,   6,5.5, 5,     4],
                   [5,1,1,14,1,1,   20,1,1,1,1,    30, 1,1,1,1,  15,1,1,10, 1,1, 36,1,  10,  5]]
    file_name = "random_pause1_{}".format(max_time)

    x_new1, y_new1, y_de1 = generate(random_pause1)
    # x_new2, y_new2, y_de2 = generate(long_pause2)
    # x_new3, y_new3, y_de3 = generate(long_pause3)
    # x_new4, y_new4, y_de4 = generate(long_pause4)
    x_new5, y_new5, y_de5 = generate_withDuration(long_pause5, max_time)
    x_new6, y_new6, y_de6 = generate_withDuration(long_pause6, max_time)
    x_new7, y_new7, y_de7 = generate_withDuration(long_pause7, max_time)

    generate_ramp_trail(15)
    # long_pause8, max_time8 = generate_random_trail()
    long_pause8 = [[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0],
                   [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]]
    long_pause8 = [[0, 3, 6, 3, 0],
                   [20, 20, 20, 20, 20]]
    max_time8 = 50
    x_new8, y_new8, y_de8 = generate_withDuration(long_pause8, max_time8, window_size=1)
    x_new = [x_new5, x_new6, x_new7]
    y_new = [y_new5, y_new6, y_new7]
    # t = np.linspace(0, 30, 2000)
    # cmd_list = []
    # for j in t:
    #     cmd_list.append(generate_random(j))
    # Plotting
    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4/2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=8, pad=0.01, length=2)
    plt.plot(x_new8, y_new8, linestyle='-', linewidth=1)
    # plt.plot(x_new8, y_de8, label='discrete derivative', linestyle='-', linewidth=2)
    # plt.plot(t, cmd_list, label='discrete derivative', linestyle='-', linewidth=2)
    # plt.legend()
    plt.xlim([-1, max_time8+1])
    plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
    plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
    plt.tight_layout()
    # plt.savefig("../figures/random_data/random_{}.png".format(np.random.rand()), dpi=600)
    # plt.savefig("./figures/longPauseInput_DataCollection/{}.png".format(file_name), dpi=600)
    # plt.show()

    plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 / 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.tick_params(labelsize=8, pad=0.01, length=2)
        plt.plot(x_new[i], y_new[i], linestyle='-', linewidth=1)
        plt.xlim([-1, max_time + 1])
        plt.xlabel('Time (s)', fontsize=8, labelpad=0.01)
        plt.ylabel('Tendon disp. (mm)', fontsize=8, labelpad=0.01)
    plt.tight_layout()
    # plt.savefig("./figures/longPauseInput_DataCollection/{}.svg".format(file_name))
    # plt.savefig("./figures/longPauseInput_DataCollection/long_pause_data2.png", dpi=600)
    # plt.show()

    label = ["Time", "Tendon1 Disp.", "Tendon2 Disp.", "Motor1 Curr.", "Motor2 Curr.", "Pos x", "Pos y", "Pos z",
             "Angle x", "Angle y", "Angle z"]
    sample_rate = 25
    pt_path = "./checkpoints/Train12345Hz1rep_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch442_best0.00011173137781125578.pt"
    # pt_path = "./checkpoints/Train12345Hz1repFakePause_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch496_best0.00012031029910321632.pt"
    pt_path = "./checkpoints/longPause_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch397_best0.0002755442473537218.pt"
    pt_path = "./checkpoints/longPause_12345Hz_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch465_best0.0002790999646156521.pt"
    # pt_path = "./checkpoints/Train12345Hz1rep_INV_LSTM_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_LSTM_L2_bs16_epoch379_best0.00019485040575563725.pt"
    # pt_path = "./checkpoints/Train12345Hz1rep_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch383_best5.044955851898722e-05.pt"
    # pt_path = "./checkpoints/Train12345Hz1rep_INV_FNN_seg50_sr25_NOrsNOfq_TCdataset_NOwd/TC_FNN_L2_bs16_epoch486_best0.00015277773349367494.pt"
    # time = np.arange(0, max_time, 1/sample_rate)
    # tendon = np.interp(time, x_new1, y_new1)
    # angle = test_LSTM(pt_path, tendon)
    # # angle = test_FNN(pt_path, tendon)
    # # angle = test_LSTM(pt_path, tendon, forward=False)
    # # angle = test_FNN(pt_path, tendon, forward=False)
    # data_to_save = np.column_stack((time, tendon, tendon, tendon, tendon, angle, angle, angle, angle, angle, angle))
    # data_to_save = np.row_stack((label, data_to_save))
    #
    # plt.figure(figsize=(88.9 / 25.4, 88.9 / 25.4 * 1.2))
    # plt.rcParams['font.family'] = 'Times New Roman'
    # ax1 = plt.subplot(2, 1, 1)
    # ax1.tick_params(labelsize=8, pad=0.01, length=2)
    # ax1.plot(time, tendon, color="red")
    # ax1.tick_params('y', colors='red')
    # ax1.set_ylim([-0.1,6.1])
    # ax1.set_xlabel("Time (s)", fontsize=8, labelpad=0.01)
    # ax1.set_ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01, color="red")
    #
    # ax2 = ax1.twinx()
    # ax2.tick_params(labelsize=8, pad=0.01, length=2)
    # ax2.plot(time, angle, color="blue")
    # ax2.set_ylim([0, 90])
    # ax2.tick_params('y', colors='blue')
    # ax2.set_ylabel("Tip angle (deg)", fontsize=8, labelpad=0.01, color="blue")
    #
    # plt.subplot(2, 1, 2)
    # plt.tick_params(labelsize=8, pad=0.01, length=2)
    # plt.plot(tendon, angle)
    # plt.ylabel("Tendon disp. (mm)", fontsize=8, labelpad=0.01)
    # plt.xlabel("Tip angle (deg)", fontsize=8, labelpad=0.01)
    # plt.tight_layout()
    # plt.savefig("./figures/New_forwardLSTM_{}.svg".format(file_name))
    # plt.show()
    # np.savetxt("./tendon_data/{}.txt".format(file_name), data_to_save, delimiter=',', fmt='%s')