import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse
import math
import numpy as np
from torch.utils.data import DataLoader
# from TC_dataset import Tendon_catheter_Dataset
from TC_dataset_tipA import Tendon_catheter_Dataset
from models.hybrid_model import LSTM_backlash_Net
from models.hybrid_model import BacklashNet
from models.hybrid_model import LSTM_backlash_sum3_Net
from models.hybrid_model import backlash_LSTM_Net
from models.hybrid_model import BacklashInverseNet
from models.hybrid_model import LSTMBacklashInverseNet
from models.hybrid_model import LSTMBacklashSumInvNet

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


def train(args, model, device, train_loader, optimizer, model_name="LSTM", seg=50, f=True, output_dim=1, invY=False):
    model.train()
    train_loss = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        if model_name == "LSTMBacklashsum":
            tendon_disp, tip_pos, freq, backlash_out = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device), data["backlash_tip"].to(device)
        else:
            tendon_disp, tip_pos, freq = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device)
        optimizer.zero_grad()

        bs = tendon_disp.shape[0]
        if hasattr(model, 'num_layers'):
            hidden = (torch.zeros(model.num_layers, bs, model.hidden_dim).to(device),
                      torch.zeros(model.num_layers, bs, model.hidden_dim).to(device))
        td = tendon_disp[:, :, 0].unsqueeze(-1)
        pre_td = tendon_disp[:, :, 1].unsqueeze(-1)
        if output_dim == 1:
            tip = tip_pos[:, :, 0].unsqueeze(-1)
            pre_tip = tip_pos[:, :, 1].unsqueeze(-1)

            if invY:
                tip = tip_pos[:, :, 1].unsqueeze(-1)
                pre_tip = tip_pos[:, :, 3].unsqueeze(-1)  # keep Y as input

            # for tipDX
            # tip_pos[:, :, [0, 2]] = 1 - (tip_pos[:, :, [0, 2]]-0.4)*3
            # tip = tip_pos[:, :, 0].unsqueeze(-1)
            # pre_tip = tip_pos[:, :, 2].unsqueeze(-1)

            # for tipDY
            # tip = tip_pos[:, :, 1].unsqueeze(-1)
            # pre_tip = tip_pos[:, :, 3].unsqueeze(-1)
        else:
            tip_pos[:, :, [0, 2]] = 1 - (tip_pos[:, :, [0, 2]]-0.4)*3
            tip = tip_pos[:, :, [0, 1]]
            pre_tip = tip_pos[:, :, [2, 3]]
        if f:
            if args.model_name == "LSTMBacklashsum":
                backlash_tip = backlash_out[:, :, 0].unsqueeze(-1)
                output, h_, lstm_out = model(td, backlash_tip, hidden)  # for lstm_backlash_sum model
            if args.model_name == "LSTMBacklash":
                output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = model(td, pre_tip, hidden)   # for LSTM-Backlash model
            if args.model_name == "Backlash":
                output, dynamic_weights, condition_lo, condition_up, condition_mid = model(td, pre_tip)    # for Backlash model
        else:
            if args.model_name == "LSTMBacklashsum":
                backlash_td = backlash_out[:, :, 0].unsqueeze(-1)
                output, h_, lstm_out = model(tip, backlash_td, hidden)  # for lstm_backlash_sum Inverse model
            if args.model_name == "Backlash":
                output, dynamic_weights = model(tip, pre_tip, pre_td)  # Backlash Inverse model
            if args.model_name == "LSTMBacklash":
                output, h_, dynamic_weights = model(tip, pre_tip, pre_td, hidden)  # LSTM-Backlash Inverse model


        # print("out shape**********", output.shape)
        output = torch.transpose(output, 1, 2)
        if f:
            if output_dim == 1:
                if invY:
                    gt = torch.transpose(tip_pos[:, :, 1:2], 1, 2)  # for Y
                else:
                    gt = torch.transpose(tip_pos[:, :, 0:1], 1, 2)  # for angle or X
                #
            else:
                gt = torch.transpose(tip_pos[:, :, [0, 1]], 1, 2)
        else:
            gt = torch.transpose(tendon_disp[:, :, 0:1], 1, 2)
        # print(output.shape, gt.shape)
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


def test(args, model, device, test_loader, model_name="LSTM", seg=50, f=True, output_dim=1, invY=False):
    model.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for b_idx, data in enumerate(test_loader):
            if model_name == "LSTMBacklashsum":
                tendon_disp, tip_pos, freq, backlash_out = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device), data["backlash_tip"].to(device)
            else:
                tendon_disp, tip_pos, freq = data["tendon_disp"].to(device), data["tip_pos"].to(device), data["freq"].to(device)
            bs = tendon_disp.shape[0]
            if hasattr(model, 'num_layers'):
                hidden = (torch.zeros(model.num_layers, bs, model.hidden_dim).to(device),
                          torch.zeros(model.num_layers, bs, model.hidden_dim).to(device))
            td = tendon_disp[:, :, 0].unsqueeze(-1)
            pre_td = tendon_disp[:, :, 1].unsqueeze(-1)
            if output_dim == 1:
                tip = tip_pos[:, :, 0].unsqueeze(-1)
                pre_tip = tip_pos[:, :, 1].unsqueeze(-1)

                if invY:
                    tip = tip_pos[:, :, 1].unsqueeze(-1)
                    pre_tip = tip_pos[:, :, 3].unsqueeze(-1)  # keep Y as input

                # for tipDX
                # tip_pos[:, :, [0, 2]] = 1 - (tip_pos[:, :, [0, 2]]-0.4)*3
                # tip = tip_pos[:, :, 0].unsqueeze(-1)
                # pre_tip = tip_pos[:, :, 2].unsqueeze(-1)

                # for tipDY
                # tip = tip_pos[:, :, 1].unsqueeze(-1)
                # pre_tip = tip_pos[:, :, 3].unsqueeze(-1)
            else:
                tip_pos[:, :, [0, 2]] = 1 - (tip_pos[:, :, [0, 2]]-0.4)*3
                tip = tip_pos[:, :, [0, 1]]
                pre_tip = tip_pos[:, :, [2, 3]]
            if f:
                if args.model_name == "LSTMBacklashsum":
                    backlash_tip = backlash_out[:, :, 0].unsqueeze(-1)
                    output, h_, lstm_out = model(td, backlash_tip, hidden)  # for lstm_backlash_sum model
                if args.model_name == "LSTMBacklash":
                    output, h_, dynamic_weights, condition_lo, condition_up, condition_mid = model(td, pre_tip, hidden)  # for LSTM-Backlash model
                if args.model_name == "Backlash":
                    output, dynamic_weights, condition_lo, condition_up, condition_mid = model(td, pre_tip)  # for Backlash model
            else:
                if args.model_name == "LSTMBacklashsum":
                    backlash_td = backlash_out[:, :, 0].unsqueeze(-1)
                    output, h_, lstm_out = model(tip, backlash_td, hidden)  # for lstm_backlash_sum Inverse model
                if args.model_name == "Backlash":
                    output, dynamic_weights = model(tip, pre_tip, pre_td)  # Backlash Inverse model
                if args.model_name == "LSTMBacklash":
                    output, h_, dynamic_weights = model(tip, pre_tip, pre_td, hidden)  # LSTM-Backlash Inverse model

            # print("test **********", output.size())
            output = torch.transpose(output, 1, 2)
            if f:
                if output_dim == 1:
                    if invY:
                        gt = torch.transpose(tip_pos[:, :, 1:2], 1, 2)  # for Y
                    else:
                        gt = torch.transpose(tip_pos[:, :, 0:1], 1, 2)  # for angle or X
                else:
                    gt = torch.transpose(tip_pos[:, :, [0, 1]], 1, 2)
            else:
                gt = torch.transpose(tendon_disp[:, :, 0:1], 1, 2)
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
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', type=str, default="LSTMBacklash")
    parser.add_argument('--checkpoints_dir', type=str,
                        default="./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25_b2/") #stdRatio1e-4_delta1e-4
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--fnn_layers', type=int, default=2)
    parser.add_argument('--std_loss_weight', type=float, default=0)
    parser.add_argument('--delta', type=float, default=1e-4, help="delta for std loss")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    # filepath = "./tendon_data/Data_with_Initialization/Sinus12"
    filepath = "./tendon_data/Data_with_Initialization/Stair5s"
    # filepath = "./tendon_data/Data_with_Initialization/SinusStair5s"
    pos = 1
    seg = 50
    sr = 25  # Hz, sampling rate
    forward = True  # True for forward kinematic modeling, False for inverse kinematic modeling
    inverseY = False  # label for training inverse model using tipY as input, OR training forward with only tipY for output
    rs = False  # random sample
    input_dim = 1  # rs and freq
    output_dim = 1  # 1 for tip angle prediction, and 2 for tip displacement prediction
    act = None
    wd = 0
    lstm_test_acc = []
    lstm_train_loss = []
    lr = 10 * 1e-4  # 10 * 1e-4
    # if "LSTMbacklash" in args.model_name:
    print('Training LSTM backlash model.')
    # model = backlash_LSTM_Net(inp_dim=input_dim, hidden_dim=64, num_layers=args.lstm_layers, act=act).to(device)
    if args.model_name == "LSTMBacklash":
        if forward:
            model = LSTM_backlash_Net(inp_dim=input_dim, hidden_dim=64, num_layers=args.lstm_layers, output_dim=output_dim, act=act).to(device)
        else:
            model = LSTMBacklashInverseNet(inp_dim=input_dim, hidden_dim=64, num_layers=args.lstm_layers, act=act).to(device)
    if args.model_name == "Backlash":
        if forward:
            model = BacklashNet().to(device)
        else:
            model = BacklashInverseNet().to(device)
    if args.model_name == "LSTMBacklashsum":
        # from backlash_dataset import Tendon_catheter_Dataset
        # from backlash_dataset_tipY import Tendon_catheter_Dataset
        if forward:
            model = LSTM_backlash_sum3_Net(inp_dim=input_dim, hidden_dim=64, num_layers=args.lstm_layers, act=act).to(device)
        else:
            model = LSTMBacklashSumInvNet(inp_dim=input_dim, hidden_dim=64, num_layers=args.lstm_layers, act=act).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr 5*1e-4 for FFN, 10*1e-4 for LSTM
    if "LSTM" in args.model_name:
        train_dataset = Tendon_catheter_Dataset("train", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs, forward=forward)
        test_dataset = Tendon_catheter_Dataset("test", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs, forward=forward)
    else:
        train_dataset = Tendon_catheter_Dataset("train", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs, forward=forward)
        test_dataset = Tendon_catheter_Dataset("test", seg=seg, filepath=filepath, pos=pos, sample_rate=sr, random_sample=rs, forward=forward)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    min_test_acc = 1000
    # checkpoint = torch.load('./checkpoints/innerTSlackSinus_LSTMBLsum_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch170.pt')
    # model.load_state_dict(checkpoint)
    for epoch in range(1, args.epochs + 1):
        print('------Train epoch---------: {} \n'.format(epoch))
        train_acc = train(args, model, device, train_data, optimizer, model_name=args.model_name, seg=seg, f=forward, output_dim=output_dim, invY=inverseY)
        test_acc = test(args, model, device, test_data, model_name=args.model_name, seg=seg, f=forward, output_dim=output_dim, invY=inverseY)
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