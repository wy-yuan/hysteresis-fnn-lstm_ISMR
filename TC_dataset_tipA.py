import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import os
from scipy.interpolate import interp1d

class Tendon_catheter_Dataset(data.Dataset):

    def __init__(self, stage, seg=50, filepath="./tendon_data/45ca/", pos=0, sample_rate=100, random_sample=False, forward=True):
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
        em_pos = data[:, 5:8]
        # tip_disp = np.linalg.norm(em_pos, axis=1)

        X = 7.98 * 25.4 - data[:, 5] + 14  # unit mm
        Y = data[:, 6] - 1.13 * 25.4  # unit mm
        # resample data using fixed sampling rate
        # sample_rate = 100  # Hz
        frequency = round(1/(time[-1]/7), 2)
        # interp_time = np.arange(0, time[-1], 1/sample_rate)
        interp_time = np.arange(10, time[-1], 1/sample_rate)
        tendon_disp_resample = np.interp(interp_time, time, tendon_disp)
        tip_A_resample = np.interp(interp_time, time, tip_A)
        tip_disp_resample = np.vstack([np.interp(interp_time, time, X), np.interp(interp_time, time, Y)]).T
        print(tip_disp_resample.shape)

        # normalization to [0, 1] and pad -1
        # tendon_disp = np.hstack([-np.ones(self.seg), tendon_disp_resample/6])
        tendon_disp = np.hstack([np.ones(self.seg)*0, tendon_disp_resample/6])
        # tendon_disp = tendon_disp_resample / 6
        # tip_A = np.hstack([-np.ones(self.seg), tip_A_resample/90])
        tip_A = np.hstack([np.ones(self.seg)*30/90, tip_A_resample/90])
        # tip_A = tip_A_resample / 90
        # tip_D = tip_disp_resample / 90
        tip_D = np.vstack([np.hstack([np.ones((self.seg, 1)) * 66 / 90, np.ones((self.seg, 1)) * 20 / 90]), tip_disp_resample / 90])
        freq = np.ones_like(tendon_disp, dtype=np.float32) * frequency
        return {'tendon_disp': tendon_disp, 'tip_A': tip_A, 'tip_D': tip_D, 'freq': freq}

    def __getitem__(self, index):
        tip_name = 'tip_A'
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
            if tip_name=="tip_A":
                tip_pos = np.vstack([self.data[data_ind][tip_name][[seq_ind + j * rs for j in range(self.seg)]],
                                         self.data[data_ind][tip_name][[seq_ind - 1 + j * rs for j in range(self.seg)]]]).T  # previous position
            else:
                tip_pos = np.hstack([self.data[data_ind][tip_name][[seq_ind + j * rs for j in range(self.seg)]],
                                     self.data[data_ind][tip_name][[seq_ind - 1 + j * rs for j in range(self.seg)]]])  # previous position
        else:
            tendon_disp = self.data[data_ind]['tendon_disp'][[seq_ind + j * rs for j in range(self.seg)]][:, np.newaxis]
            if tip_name == "tip_A":
                tip_pos = self.data[data_ind][tip_name][[seq_ind+j*rs for j in range(self.seg)]][:, np.newaxis]
            else:
                tip_pos = self.data[data_ind][tip_name][[seq_ind + j * rs for j in range(self.seg)]]
        # print(tip_pos.shape)
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
    data = Tendon_catheter_Dataset("test")




