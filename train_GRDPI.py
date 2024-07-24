import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse
import math
import numpy as np
from torch.utils.data import DataLoader
from TC_dataset_tipA import Tendon_catheter_Dataset
from models.GRDPI_model import GRDPINet
from models.GRDPI_model import GRDPI_inv_Net

import os
import time
criterionMSE = nn.MSELoss()

def std_dev(input, output, delta=1e-4):
    # print("input shape ", input.shape)
    # print("output shape ", output.shape)
    # std_loss = torch.mean(torch.std(output, dim=2) - torch.std(input, dim=1))
    std_loss = torch.mean(torch.std(output, dim=2) / (torch.std(input, dim=1) + delta))
    # print(std_loss)
    return std_loss


def backlash_weight_loss(dynamic_weights, alpha=0.0):
    # Calculate the differences between consecutive timesteps
    timestep_diffs = dynamic_weights[:, 1:, :] - dynamic_weights[:, :-1, :]
    # Calculate the L2 norm (squared) of the differences
    regularization_term = torch.norm(timestep_diffs, p=2, dim=-1) ** 2
    # Sum over the sequence and average over the batch
    loss = alpha * torch.mean(regularization_term)
    # loss = alpha * torch.mean(torch.std(dynamic_weights, dim=1))
    return loss

def direction_loss(output, poses, pre_tip, pre_td, td, epsilon=1e-8):
    pre_tip = torch.transpose(pre_tip, 1, 2)
    pre_td = torch.transpose(pre_td, 1, 2)
    td = torch.transpose(td, 1, 2)
    # print(output.shape, poses.shape, pre_tip.shape, pre_td.shape, td.shape)
    x1 = td - pre_td
    x2 = td - pre_td
    y1 = output - pre_tip
    y2 = poses - pre_tip
    dot_product = x1*x2 + y1*y2
    norm_a = torch.norm(torch.cat((x1, y1), dim=1), dim=1).unsqueeze(1)
    norm_b = torch.norm(torch.cat((x2, y2), dim=1), dim=1).unsqueeze(1)
    # print(dot_product.shape, norm_a.shape, norm_b.shape)
    cos_angle = dot_product / (norm_a * norm_b + epsilon)
    ang_loss = torch.mean(torch.acos(cos_angle))
    # print(cos_angle)
    return ang_loss


def train(args, model, device, train_loader, optimizer, model_name="LSTM", seg=50, f=True):
    model.train()
    train_loss = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        # tendon_disp, tip_pos, freq, backlash_out = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device), data["backlash_tip"].to(device)
        tendon_disp, tip_pos, freq = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device)
        optimizer.zero_grad()
        # tendon_disp = tendon_disp
        # tip_pos = tip_pos
        bs = tendon_disp.shape[0]
        td = tendon_disp[:, :, 0].unsqueeze(-1)
        pre_td = tendon_disp[:, :, 1].unsqueeze(-1)
        tip = tip_pos[:, :, 0].unsqueeze(-1)
        pre_tip = tip_pos[:, :, 1].unsqueeze(-1)
        # print("h_init.shape:", h_init.shape)
        if f:
            h_init = td[:, 0:1, :].expand(td.shape[0], 1, 4)
            output, dynamic_weights, h_ = model(td, pre_td, h_init)    # for
        else:
            h_init = tip[:, 0:1, :].expand(tip.shape[0], 1, 4)
            output, dynamic_weights, h_ = model(tip, pre_tip, h_init)    #


        # print("out shape**********", output.shape)
        output = torch.transpose(output, 1, 2)
        if f:
            gt = torch.transpose(tip_pos[:, :, 0:1], 1, 2)
            # gt = torch.transpose(tip_pos[:, :, 1:2], 1, 2)
        else:
            gt = torch.transpose(tendon_disp[:, :, 0:1], 1, 2)

        # loss last number
        # output = output[:,:,-1]
        # gt = gt[:,:,-1]

        if args.std_loss_weight > 0:
            std_loss = std_dev(tendon_disp, output, delta=args.delta) * args.std_loss_weight
            # print("std loss", std_loss)
            loss = criterionMSE(output, gt)*loss_weight + std_loss
        else:
            # bl_loss = backlash_weight_loss(dynamic_weights)
            loss = criterionMSE(output, gt)*loss_weight
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += 1
        if batch_idx % args.log_interval == 0:
            # print("bl loss", bl_loss)
            print('Train Batch: {} \tLoss: {:.6f}'.format(
                batch_idx, loss.item()))
        # if batch_idx % 500 == 0:
        #     print(dynamic_weights)
    return train_loss/total


def test(args, model, device, test_loader, model_name="LSTM", seg=50, f=True):
    model.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for b_idx, data in enumerate(test_loader):
            # tendon_disp, tip_pos, freq, backlash_out = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device), data["backlash_tip"].to(device)
            tendon_disp, tip_pos, freq = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device)
            bs = tendon_disp.shape[0]
            # tendon_disp = tendon_disp
            # tip_pos = tip_pos
            # tip_pos = (tip_pos - 0.33) * 1.5
            # h_init = torch.zeros(bs, 1, 1).to(device)
            td = tendon_disp[:, :, 0].unsqueeze(-1)
            pre_td = tendon_disp[:, :, 1].unsqueeze(-1)
            tip = tip_pos[:, :, 0].unsqueeze(-1)
            pre_tip = tip_pos[:, :, 1].unsqueeze(-1)
            if f:
                h_init = td[:, 0:1, :].expand(td.shape[0], 1, 4)
                output, dynamic_weights, h_ = model(td, pre_td, h_init)    #
            else:
                h_init = tip[:, 0:1, :].expand(tip.shape[0], 1, 4)
                output, dynamic_weights, h_ = model(tip, pre_tip, h_init)    #

            # print("test **********", output.size())
            output = torch.transpose(output, 1, 2)
            if f:
                gt = torch.transpose(tip_pos[:, :, 0:1], 1, 2)
                # gt = torch.transpose(tip_pos[:, :, 1:2], 1, 2)
            else:
                gt = torch.transpose(tendon_disp[:, :, 0:1], 1, 2)

            # loss last number
            # output = output[:, :, -1]
            # gt = gt[:, :, -1]

            if args.std_loss_weight > 0:
                std_loss = std_dev(tendon_disp, output, delta=args.delta) * args.std_loss_weight
                loss = criterionMSE(output, gt) * loss_weight + std_loss
            else:
                # bl_loss = backlash_weight_loss(dynamic_weights)
                loss = criterionMSE(output, gt) * loss_weight
                # loss = criterionMSE(output, poses) * loss_weight + 0.01*direction_loss(output, poses, pre_tip, pre_td, td)
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
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', type=str, default="GRDPI")
    parser.add_argument('--checkpoints_dir', type=str,
                        default="./checkpoints/Sinus12_GRDPIInv_seg50_sr25_relu_noGamma_a4_StdLoss/") #stdRatio1e-4_delta1e-4
    parser.add_argument('--std_loss_weight', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=1e-4, help="delta for std loss")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    filepath = "./tendon_data/Data_with_Initialization/Sinus12"
    # filepath = "./tendon_data/Data_with_Initialization/Stair5s"
    pos = 1
    seg = 50
    sr = 25  # Hz, sampling rate
    forward = False  # True for forward kinematic modeling, False for inverse kinematic modeling
    rs = False  # random sample
    input_dim = 1  # rs and freq
    act = None
    wd = 0
    lstm_test_acc = []
    lstm_train_loss = []
    lr = 10 * 1e-4  # 10 * 1e-4
    # if "LSTMbacklash" in args.model_name:
    print('Training rate-dependent P-I model.')
    if forward:
        model = GRDPINet().to(device)
    else:
        model = GRDPI_inv_Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
    # checkpoint = torch.load('./checkpoints/Sinus_GRDPI_seg50_sr25_relu_onlya1/TC_GRDPI_bs16_epoch4_best0.00074711664416038.pt')
    # model.load_state_dict(checkpoint)
    for epoch in range(1, args.epochs + 1):
        print('------Train epoch---------: {} \n'.format(epoch))
        train_acc = train(args, model, device, train_data, optimizer, model_name=args.model_name, seg=seg, f=forward)
        test_acc = test(args, model, device, test_data, model_name=args.model_name, seg=seg, f=forward)
        print('\n--Train set: Average loss: {:.6f}\n'.format(train_acc))
        lstm_test_acc.append(test_acc)
        lstm_train_loss.append(train_acc)
        if args.save_model:
            if epoch % 2 == 0:
                torch.save(model.state_dict(), args.checkpoints_dir +
                           "TC_" + args.model_name + "_bs{}_epoch{}.pt".format(str(args.batch_size), str(epoch)))
            if epoch > 2 and test_acc < min_test_acc:
                torch.save(model.state_dict(), args.checkpoints_dir +
                           "TC_" + args.model_name + "_bs{}_epoch{}_best{}.pt".format(str(args.batch_size),
                                                                                            str(epoch), str(test_acc)))
                min_test_acc = test_acc

    print(model)
    pickle.dump( {'test': lstm_test_acc, 'train': lstm_train_loss},
                 open(args.checkpoints_dir + "TC_" + args.model_name + "_acc_bs{}_epoch{}.pkl".format(str(args.batch_size), str(args.epochs)), "wb"))

if __name__ == '__main__':
    torch.cuda.synchronize()
    start_time = time.time()
    loss_weight = 1
    main()
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    print("Model training time in total: ", total_time)