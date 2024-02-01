import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse
import math
import numpy as np
from torch.utils.data import DataLoader
# from tendon_catheter_dataset import Tendon_catheter_Dataset
from TC_dataset import Tendon_catheter_Dataset
import os
import time
criterionMSE = nn.MSELoss()

class LSTMNet(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        # nn.init.orthogonal(self.rnn.weight_ih_l0)
        # nn.init.orthogonal(self.rnn.weight_hh_l0)

    def forward(self, points, hidden):
        lstm_out, h_ = self.rnn(points, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)
        # out = F.log_softmax(linear_out, dim=1)
        return linear_out, h_

class FFNet(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, seg=50, dropout=0, num_layers=2):
        super(FFNet, self).__init__()
        # self.num_layers = 5
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc1 = nn.Linear(inp_dim*seg, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # two hidden layers
        self.fc3 = nn.Linear(hidden_dim, self.output_dim*1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        if self.num_layers == 2:
            x = F.relu(self.fc2(x))
        out = self.fc3(x)
        out = out.unsqueeze(dim=2)
        return out

def std_dev(input, output, delta=1e-4):
    # print("input shape ", input.shape)
    # print("output shape ", output.shape)
    # std_loss = torch.mean(torch.std(output, dim=2) - torch.std(input, dim=1))
    std_loss = torch.mean(torch.std(output, dim=2) / (torch.std(input, dim=1) + delta))
    # print(std_loss)
    return std_loss

def train(args, model, device, train_loader, optimizer, model_name="LSTM", seg=50, f=True, fq=True):
    model.train()
    train_loss = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        if f:
            tendon_disp, tip_pos, freq = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device)
        else:
            # swap the input and output for inverse kinematic modeling
            tendon_disp, tip_pos, freq = data["tip_pos"].to(device), data["tendon_disp"].to(device), data["freq"].to(device)
        if fq:
            tendon_disp = torch.cat((tendon_disp, freq), dim=2)
        optimizer.zero_grad()

        bs = tendon_disp.shape[0]
        if model_name == "LSTM":
            hidden = (torch.zeros(model.num_layers, bs, model.hidden_dim).to(device),
                      torch.zeros(model.num_layers, bs, model.hidden_dim).to(device))
            output, h_ = model(tendon_disp, hidden)
        else:
            output = model(tendon_disp)
        # print("out shape**********", output.shape)
        output = torch.transpose(output, 1, 2)
        # poses shape******* torch.Size([16, 50, 1])
        # print(tip_pos.shape)
        if model_name == "FNN":
            poses = torch.transpose(tip_pos[:, seg-1:seg, 0:1], 1, 2)
        else:
            poses = torch.transpose(tip_pos[:, :, 0:1], 1, 2)
            # output = torch.transpose(output[: 49:50, 0:1], 1, 2)
            # poses = torch.transpose(tip_pos[: 49:50, 0:1], 1, 2)
        # print("output shape", output.shape, "poses shape", poses.shape)
        # loss = F.pairwise_distance(output[:, :, :], poses[:, :, :], p=2)
        # loss = torch.mean(loss)
        if args.std_loss_weight > 0:
            std_loss = std_dev(tendon_disp, output, delta=args.delta)*args.std_loss_weight
            # print("std loss", std_loss)
            loss = criterionMSE(output, poses)*loss_weight + std_loss
        else:
            loss = criterionMSE(output, poses)*loss_weight
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        train_loss += loss.item()
        total += 1
        if batch_idx % args.log_interval == 0:
            print('Train Batch: {} \tLoss: {:.6f}'.format(
                batch_idx, loss.item()))
    return train_loss/total


def test(args, model, device, test_loader, model_name="LSTM", seg=50, f=True, fq=True):
    model.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for b_idx, data in enumerate(test_loader):
            if f:
                tendon_disp, tip_pos, freq = data["tendon_disp"].to(device), data["tip_pos"].to(device), data[
                    "freq"].to(device)
            else:
                # swap the input and output for inverse kinematic modeling
                tendon_disp, tip_pos, freq = data["tip_pos"].to(device), data["tendon_disp"].to(device), data[
                    "freq"].to(device)
            if fq:
                tendon_disp = torch.cat((tendon_disp, freq), dim=2)
            bs = tendon_disp.shape[0]
            if model_name == "LSTM":
                hidden = (torch.zeros(model.num_layers, bs, model.hidden_dim).to(device),
                          torch.zeros(model.num_layers, bs, model.hidden_dim).to(device))
                output, h_ = model(tendon_disp, hidden)
            else:
                output = model(tendon_disp)
            # print("test **********", output.size())
            output = torch.transpose(output, 1, 2)
            if model_name == "FNN":
                poses = torch.transpose(tip_pos[:, seg-1:seg, 0:1], 1, 2)
            else:
                poses = torch.transpose(tip_pos[:, :, 0:1], 1, 2)
                # output = torch.transpose(output[: 49:50, 0:1], 1, 2)
                # poses = torch.transpose(tip_pos[: 49:50, 0:1], 1, 2)
            # poses = torch.transpose(tip_pos, 1, 2)
            # print(poses.shape)
            # loss = F.pairwise_distance(output[:,:,:], poses[:,:,:], p=2)
            # loss = torch.mean(loss)
            if args.std_loss_weight > 0:
                std_loss = std_dev(tendon_disp, output, delta=args.delta) * args.std_loss_weight
                loss = criterionMSE(output, poses) * loss_weight + std_loss
            else:
                loss = criterionMSE(output, poses) * loss_weight
            test_loss += loss.item()
            total += 1
    test_loss /= total

    print('\n--Test set: Average loss: {:.6f}\n'.format(
        test_loss))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', type=str, default="LSTM")
    parser.add_argument('--checkpoints_dir', type=str,
                        default="./checkpoints/innerTSlack40_LSTM_layer2_seg50_sr25_stdRatio1e-4_delta1e-4/") #stdRatio1e-4_delta1e-4
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--fnn_layers', type=int, default=2)
    parser.add_argument('--std_loss_weight', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=1e-4, help="delta for std loss")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    filepath = "./tendon_data/innerTSlack40"
    pos = 0
    seg = 50
    sr = 25  # Hz, sampling rate
    forward = True  # True for forward kinematic modeling, False for inverse kinematic modeling
    rs = False  # random sample
    fq = False
    input_dim = 1  # rs and freq
    act = None
    wd = 0
    lstm_test_acc = []
    lstm_train_loss = []
    if "LSTM" in args.model_name:
        print('Training LSTM.')
        model = LSTMNet(inp_dim=input_dim, hidden_dim=64, num_layers=args.lstm_layers, act=act).to(device)
        lr = 10 * 1e-4  # 10 * 1e-4
    else:
        print('Training FNN.')
        model = FFNet(inp_dim=input_dim, hidden_dim=64, seg=seg, num_layers=args.fnn_layers).to(device)
        lr = 10 * 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr 5*1e-4 for FFN, 10*1e-4 for LSTM
    if "LSTM" in args.model_name:
        train_dataset = Tendon_catheter_Dataset("train", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs)
        test_dataset = Tendon_catheter_Dataset("test", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs)
    else:
        train_dataset = Tendon_catheter_Dataset("train", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs)
        test_dataset = Tendon_catheter_Dataset("test", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    min_test_acc = 1000
    for epoch in range(1, args.epochs + 1):
        print('------Train epoch---------: {} \n'.format(epoch))
        train_acc = train(args, model, device, train_data, optimizer, model_name=args.model_name, seg=seg, f=forward, fq=fq)
        test_acc = test(args, model, device, test_data, model_name=args.model_name, seg=seg, f=forward, fq=fq)
        print('\n--Train set: Average loss: {:.6f}\n'.format(train_acc))
        lstm_test_acc.append(test_acc)
        lstm_train_loss.append(train_acc)
        if args.save_model:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), args.checkpoints_dir +
                           "TC_" + args.model_name + "_L{}_bs{}_epoch{}.pt".format(str(args.lstm_layers),
                                                                                     str(args.batch_size), str(epoch)))
            if epoch > 10 and test_acc < min_test_acc:
                torch.save(model.state_dict(), args.checkpoints_dir +
                           "TC_" + args.model_name + "_L{}_bs{}_epoch{}_best{}.pt".format(str(args.lstm_layers),
                                                                                            str(args.batch_size),
                                                                                            str(epoch), str(test_acc)))
                min_test_acc = test_acc

    print(model)
    pickle.dump( {'test': lstm_test_acc, 'train': lstm_train_loss},
                 open(args.checkpoints_dir + "TC_" + args.model_name + "_L{}_acc_bs{}_epoch{}.pkl".format(
                     str(args.lstm_layers), str(args.batch_size), str(args.epochs)), "wb"))

if __name__ == '__main__':
    torch.cuda.synchronize()
    start_time = time.time()
    loss_weight = 1
    main()
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    print("Model training time in total: ", total_time)