import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import os
from scipy.interpolate import interp1d
from models.hybrid_model import fixedBacklashTipY
from models.hybrid_model import fixedBacklashTipYInverseNet
import copy

def data_filter(data, datatype="angle"):
    if datatype == "angle":
        # 小范围的上升或下降，就把这个值拉到和前一个值相同
        filtered = copy.deepcopy(data)
        for i in range(1, data.shape[0]-1):
            v = abs(data[i]-filtered[i-1])
            if v < 0.0003:  # v should be larger than 0.0001,
                filtered[i] = filtered[i-1]
    elif datatype == "disp":
        filtered1 = copy.deepcopy(data)
        n = 1
        for i in range(0, data.shape[0]):
            filtered1[i] = np.mean(data[max(i-n, 0):i+n+1])
        filtered = copy.deepcopy(filtered1)
        for i in range(1, data.shape[0] - 1):
            v = abs(data[i] - filtered[i - 1])
            if v < 0.0015:  # v should be larger than 0.0001,
                filtered[i] = filtered[i - 1]
        for j in range(0):
            filtered2 = copy.deepcopy(filtered)
            n = 1
            for i in range(0, data.shape[0]):
                arr = filtered[max(i - n, 0):i + n + 1]
                if len(arr)<2*n+1:
                    filtered2[i] = np.mean(arr)
                else:
                    filtered2[i] = np.sum(arr*np.array([1,2,1])/4)
    return filtered

def test_backlash(data, forward=True):
    device = "cuda"
    if forward:
        model = fixedBacklashTipY(datatype="stair")
    else:
        model = fixedBacklashTipYInverseNet(datatype="stair")
    model.cuda()
    model.eval()

    tip_name = "tip_D"
    data[tip_name][:, 1] = data_filter(data[tip_name][:, 1], datatype="disp")

    joints = data['tendon_disp'][:, np.newaxis].astype("float32")
    pre_pos = data[tip_name][0:1, 1:2].astype("float32")  # for tipY only
    tips = data[tip_name][:, 1:2].astype("float32")  # for tipY only
    pre_td = data['tendon_disp'][0:1, np.newaxis].astype("float32")

    out = []
    for i in range(0, data['tendon_disp'].shape[0]):
        joint = joints[i:i + 1, 0:1]
        tip = tips[i:i + 1, 0:1]
        if forward:
            output = model(torch.tensor([joint]).to(device), torch.tensor([pre_pos]).to(device))
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        else:
            output = model(torch.tensor([tip]).to(device), torch.tensor([pre_pos]).to(device), torch.tensor([pre_td]).to(device))  # x_curr, _x_prev, u_pre
            pre_td = output.detach().cpu().numpy()[0]
            out.append(pre_td[0])
            pre_pos = tip
    out = np.array(out)
    output = out[:, 0]
    return output


class Tendon_catheter_Dataset(data.Dataset):

    def __init__(self, stage, seg=50, filepath="./tendon_data/45ca/", pos=0, sample_rate=100, random_sample=False, forward=True):
        self.stage = stage
        self.seg = seg
        self.data = []
        self.pos = pos
        self.random_sample = random_sample
        self.forward = forward
        if stage == "train":
            train_path = os.path.join(filepath, "train")
            for i, name in enumerate(os.listdir(train_path)):
                data_path = os.path.join(train_path, name)
                data = self.load_data(data_path, sample_rate=sample_rate)
                data['backlash_tip'] = test_backlash(data, forward=forward)
                self.data.append(data)

        elif stage == "test":
            test_path = os.path.join(filepath, "validate")
            for i, name in enumerate(os.listdir(test_path)):
                data_path = os.path.join(test_path, name)
                data = self.load_data(data_path, sample_rate=sample_rate)
                data['backlash_tip'] = test_backlash(data, forward=forward)
                self.data.append(data)

        number = 0
        self.index_list = []
        for i in range(len(self.data)):
            number += self.data[i]['tendon_disp'].shape[0]
            self.index_list = self.index_list + list(np.ones(self.data[i]['tendon_disp'].shape[0], dtype=np.int8)*i)

        if stage == "train":
            self.number = number
        elif stage == "test":
            self.number = int(number/5)

    def load_data(self, data_path, sample_rate=100):
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        time = data[:, 0]
        tendon_disp = data[:, 1]
        tip_A = data[:, 8]
        # resample data using fixed sampling rate
        # sample_rate = 100  # Hz
        X = 7.98 * 25.4 - data[:, 5] + 14  # unit mm
        Y = data[:, 6] - 1.13 * 25.4  # unit mm
        frequency = round(1/(time[-1]/7), 2)
        # interp_time = np.arange(0, time[-1], 1/sample_rate)
        interp_time = np.arange(10, time[-1], 1/sample_rate)
        tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
        tip_A_resample = np.interp(interp_time, time, tip_A)
        tip_disp_resample = np.vstack([np.interp(interp_time, time, X), np.interp(interp_time, time, Y)]).T

        # normalization to [0, 1] and pad -1
        tendon_disp = np.hstack([np.ones(self.seg)*0, tendon_disp_resample/6])
        tip_A = np.hstack([np.ones(self.seg)*30/90, tip_A_resample/90])
        tip_D = np.vstack([np.hstack([np.ones((self.seg, 1)) * 66 / 90, np.ones((self.seg, 1)) * 20 / 90]), tip_disp_resample / 90])
        freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
        return {'tendon_disp': tendon_disp, 'tip_A': tip_A, 'tip_D': tip_D, 'freq': freq}

    def __getitem__(self, index):
        # tendon_disp = self.data[index][:, [1]]
        # tip_pos = self.data[index][:, 3:6]
        if self.random_sample:
            rs = np.random.randint(1, 10, 1)[0]
        else:
            rs = 1
        # data_ind = np.random.randint(0, len(self.data), 1)[0]
        data_ind = np.random.choice(self.index_list)
        seq_ind = np.random.randint(1, self.data[data_ind]['tendon_disp'].shape[0] - self.seg*rs, 1)[0]
        if self.pos == 1:
            tendon_disp = np.vstack([self.data[data_ind]['tendon_disp'][[seq_ind+j*rs for j in range(self.seg)]],
                                     self.data[data_ind]['tendon_disp'][[seq_ind-1+j*rs for j in range(self.seg)]]]).T  # previous position
            backlash_tip = np.vstack([self.data[data_ind]['backlash_tip'][[seq_ind + j * rs for j in range(self.seg)]],
                                     self.data[data_ind]['backlash_tip'][[seq_ind - 1 + j * rs for j in range(self.seg)]]]).T  # previous position
            tip_pos = np.hstack([self.data[data_ind]['tip_D'][[seq_ind + j * rs for j in range(self.seg)]],
                                 self.data[data_ind]['tip_D'][
                                     [seq_ind - 1 + j * rs for j in range(self.seg)]]])  # previous position
        else:
            tendon_disp = self.data[data_ind]['tendon_disp'][[seq_ind + j * rs for j in range(self.seg)]][:, np.newaxis]
            backlash_tip = self.data[data_ind]['backlash_tip'][[seq_ind + j * rs for j in range(self.seg)]][:, np.newaxis]
        # print(tendon_disp.shape)
            tip_pos = self.data[data_ind]['tip_D'][[seq_ind + j * rs for j in range(self.seg)]]
        freq = self.data[data_ind]['freq'][[seq_ind + j * rs for j in range(self.seg)]][:, np.newaxis]

        tendon_disp = torch.Tensor(tendon_disp).type(torch.float)
        backlash_tip = torch.Tensor(backlash_tip).type(torch.float)
        tip_pos = torch.Tensor(tip_pos).type(torch.float)

        return {'tendon_disp': tendon_disp, 'tip_pos': tip_pos, 'freq': freq, 'backlash_tip': backlash_tip}

    def __len__(self):
        """Return the total number of samples
        """

        # data_len = int(number)
        data_len = self.number
        return data_len




