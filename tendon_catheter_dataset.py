import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import os
from scipy.interpolate import interp1d

class Tendon_catheter_Dataset(data.Dataset):

    def __init__(self, stage, seg=50, filepath="./tendon_data/45ca/", pos=0, sample_rate=100, random_sample=False):
        self.stage = stage
        self.seg = seg
        self.data = []
        self.pos = pos
        self.random_sample = random_sample
        if stage == "train":
            train_path = os.path.join(filepath, "train")
            for i, name in enumerate(os.listdir(train_path)):
                data_path = os.path.join(train_path, name)
                data = self.load_data(data_path, sample_rate=sample_rate)
                self.data.append(data)

        elif stage == "test":
            test_path = os.path.join(filepath, "validate")
            for i, name in enumerate(os.listdir(test_path)):
                data_path = os.path.join(test_path, name)
                data = self.load_data(data_path, sample_rate=sample_rate)
                self.data.append(data)

        number = 0
        for i in range(len(self.data)):
            number += self.data[i]['tendon_disp'].shape[0]

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
        frequency = round(1/(time[-1]/7), 2)
        interp_time = np.arange(0, time[-1], 1/sample_rate)
        tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
        tip_A_resample = np.interp(interp_time, time, tip_A)

        # normalization to [0, 1] and pad -1
        tendon_disp = np.hstack([-np.ones(self.seg), tendon_disp_resample/6])
        tip_A = np.hstack([-np.ones(self.seg), tip_A_resample/90])
        freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
        return {'tendon_disp': tendon_disp, 'tip_A': tip_A, 'freq': freq}

    def __getitem__(self, index):
        # tendon_disp = self.data[index][:, [1]]
        # tip_pos = self.data[index][:, 3:6]
        if self.random_sample:
            rs = np.random.randint(1, 10, 1)[0]
        else:
            rs = 1
        data_ind = np.random.randint(0, len(self.data), 1)[0]
        seq_ind = np.random.randint(1, self.data[data_ind]['tendon_disp'].shape[0] - self.seg*rs, 1)[0]
        if self.pos == 1:
            tendon_disp = np.hstack([self.data[data_ind]['tendon_disp'][[seq_ind+j*rs for j in range(self.seg)]],
                                     self.data[data_ind]['tendon_disp'][[seq_ind-1+j*rs for j in range(self.seg)]]])   # previous position
        else:
            tendon_disp = self.data[data_ind]['tendon_disp'][[seq_ind + j * rs for j in range(self.seg)]][:, np.newaxis]
        # print(tendon_disp.shape)
        tip_pos = self.data[data_ind]['tip_A'][[seq_ind+j*rs for j in range(self.seg)]][:, np.newaxis]
        freq = self.data[data_ind]['freq'][[seq_ind + j * rs for j in range(self.seg)]][:, np.newaxis]

        tendon_disp = torch.Tensor(tendon_disp).type(torch.float)
        tip_pos = torch.Tensor(tip_pos).type(torch.float)

        return {'tendon_disp': tendon_disp, 'tip_pos': tip_pos, 'freq': freq}

    def __len__(self):
        """Return the total number of samples
        """

        # data_len = int(number)
        data_len = self.number
        return data_len

if __name__ == '__main__':
    # make fake data for testing
    data_path = r'./tendon_data/45ca/train/0_5hz_rep3.txt'
    out_path = r'./tendon_data/45ca_fakedata/train/0_4hz_rep3.txt'
    header_list = ["Time", "Tendon1 Disp.", "Tendon2 Disp.", "Motor1 Curr.", "Motor2 Curr.", "Pos x", "Pos y", "Pos z",
                    "Angle x", "Angle y", "Angle z"]
    data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    data[:, 0] = data[:, 0] * 5 / 4
    np.savetxt(out_path, data, delimiter=',', fmt='%.14f', header=','.join(header_list), )




