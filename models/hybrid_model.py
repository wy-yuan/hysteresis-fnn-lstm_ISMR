import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def soft_clamp(x, min_val, max_val):
    """
    Softly clamp the values of x between min_val and max_val using a sigmoid function.

    Args:
        x (torch.Tensor): The input tensor.
        min_val (float): The minimum value for clamping.
        max_val (float): The maximum value for clamping.

    Returns:
        torch.Tensor: The softly clamped tensor.
    """
    scaled_x = torch.tanh(x)
    # Scale and shift to fit the desired range
    transformed_x = (scaled_x + 1) / 2 * (max_val - min_val) + min_val
    # clamped_x = torch.sigmoid(x) * (max_val - min_val) + min_val
    return transformed_x


class Backlash(nn.Module):
    def __init__(self):
        super(Backlash, self).__init__()

    def h(self, s):
        """Step function, see (4) in [1]"""
        s_copy = s.clone()
        s_copy[s > 0] = 0
        s_copy[s <= 0] = 1
        return s_copy

    def OneStep_forward(self, x, _out_prev, weights):
        m_lo, m_up, c_lo, c_up = weights[:, :, 0].unsqueeze(-1), weights[:, :, 1].unsqueeze(-1), weights[:, :,
                                 2].unsqueeze(-1), weights[:, :, 3].unsqueeze(-1)
        f1 = self.h((m_lo.detach() * x + m_lo.detach() * c_lo.detach() - _out_prev) / m_lo.detach())
        f2 = self.h((_out_prev - m_up.detach() * x - m_up.detach() * c_up.detach()) / m_up.detach())
        # f1 = self.h((m_lo * x + m_lo * c_lo - _out_prev) / m_lo)
        # f2 = self.h((_out_prev - m_up * x - m_up * c_up) / m_up)
        out_curr = m_lo * x * f1 + m_lo * c_lo * f1 + m_up * x * f2 + m_up * c_up * f2 + _out_prev * (1 - f1) * (1 - f2)
        return out_curr

    def forward(self, x, _out_prev, lstm_outs):
        pre_out_i = _out_prev[:, 0:1, :]
        custom_output = torch.empty(x.shape[0], 0, 1).to("cuda")
        for i in range(x.shape[1]):
            pre_out_i = self.OneStep_forward(x[:, i:i + 1, :], pre_out_i, lstm_outs[:, i:i + 1, :])
            custom_output = torch.cat((custom_output, pre_out_i), dim=1)

        condition_lo, condition_up, condition_mid = 0, 0, 0
        return custom_output, lstm_outs, condition_lo, condition_up, condition_mid

    def forward_v1(self, x, _out_prev, lstm_outs):
        # x, _out_prev have shapes [batch size, sequence length, feature size]
        # lstm_outs has shape [batch size, sequence length, 4]
        bs, seq = lstm_outs.shape[0], lstm_outs.shape[1]
        # add boundary conditions
        device = "cuda"
        lower_bounds = torch.tensor([-1, -1, -1, -1]).to(device)
        upper_bounds = torch.tensor([1, 1, 1, 1]).to(device)
        lower_bounds_expanded = lower_bounds.unsqueeze(0).unsqueeze(0).expand(bs, seq, -1)
        upper_bounds_expanded = upper_bounds.unsqueeze(0).unsqueeze(0).expand(bs, seq, -1)
        # dynamic_weights = torch.clamp(lstm_outs, min=lower_bounds_expanded, max=upper_bounds_expanded)
        dynamic_weights = soft_clamp(lstm_outs, min_val=lower_bounds_expanded, max_val=upper_bounds_expanded)

        # Expand dynamic weights for broadcasting
        m_lo = dynamic_weights[:, :, 0].unsqueeze(-1)  # Shape [batch size, sequence length, 1]
        m_up = dynamic_weights[:, :, 1].unsqueeze(-1)
        c_lo = dynamic_weights[:, :, 2].unsqueeze(-1)
        c_up = dynamic_weights[:, :, 3].unsqueeze(-1)

        # Calculate z_lo and z_up
        z_lo = _out_prev / m_lo - c_lo
        z_up = _out_prev / m_up - c_up

        # Initialize output tensor
        out_curr = torch.zeros_like(x)

        # Condition masks
        condition_lo = (x <= z_lo)
        condition_up = (x >= z_up)
        condition_mid = (~condition_lo & ~condition_up)

        # Apply piecewise function using conditions
        out_curr_lo = m_lo * (x + c_lo)
        out_curr_up = m_up * (x + c_up)

        out_curr = torch.where(condition_lo, out_curr_lo, out_curr)
        out_curr = torch.where(condition_up, out_curr_up, out_curr)
        out_curr = torch.where(condition_mid, _out_prev, out_curr)

        return out_curr, dynamic_weights, condition_lo, condition_up, condition_mid


class LSTM_backlash_Net(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0, act=None):
        super(LSTM_backlash_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 4*output_dim  # for 4 parameters in backlash model
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        self.custom_layer = Backlash()
        if self.output_dim/4 == 2:
            self.custom_layer2 = Backlash()

    def forward(self, points, pre_points, hidden):
        # points: current input
        # pre_points: previous output
        lstm_out, h_ = self.rnn(points, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)

        if self.output_dim/4 == 1:
            custom_output, dynamic_weights, condition_lo, condition_up, condition_mid = self.custom_layer(points, pre_points, linear_out)
        elif self.output_dim/4 == 2:
            custom_output1, dynamic_weights, condition_lo, condition_up, condition_mid = self.custom_layer(points, pre_points[:, :, 0:1], linear_out[:, :, :4])
            custom_output2, dynamic_weights2, condition_lo, condition_up, condition_mid = self.custom_layer2(points, pre_points[:, :, 1:2], linear_out[:, :, 4:])
            custom_output = torch.cat((custom_output1, custom_output2), dim=2)
        return custom_output, h_, dynamic_weights, condition_lo, condition_up, condition_mid


class BacklashNet(nn.Module):
    def __init__(self):
        super(BacklashNet, self).__init__()
        self.weights = nn.Parameter(torch.randn(4))

    def h(self, s):
        """Step function, see (4) in [1]"""
        s_copy = s.clone()
        s_copy[s > 0] = 0
        s_copy[s <= 0] = 1
        return s_copy

    def OneStep_forward(self, x, _out_prev):
        m_lo, m_up, c_lo, c_up = self.weights[0], self.weights[1], self.weights[2], self.weights[3]
        # 10.4949016, 10.69768651, 0.47326037, -0.16193161
        # y = (m/15) * (x + c/6+55/m/6-0.5)
        # (10.4949016/15) = 0.6996601066666667 , (0.47326037/6+55/10.4949016/6-0.5 = 0.45231671053065725
        # (10.69768651/15) =  0.7131791006666667, -0.16193161/6+55/10.69768651/6-0.5 = 0.329894487784294
        # m_lo, m_up, c_lo, c_up = torch.tensor(0.70),  torch.tensor(0.71),  torch.tensor(0.45),  torch.tensor(0.33)
        f1 = self.h((m_lo.detach() * x + m_lo.detach() * c_lo.detach() - _out_prev) / m_lo.detach())
        f2 = self.h((_out_prev - m_up.detach() * x - m_up.detach() * c_up.detach()) / m_up.detach())
        out_curr = m_lo * x * f1 + m_lo * c_lo * f1 + m_up * x * f2 + m_up * c_up * f2 + _out_prev * (1 - f1) * (1 - f2)
        return out_curr

    def forward(self, x, _out_prev):
        pre_out_i = _out_prev[:, 0:1, :]
        custom_output = torch.empty(x.shape[0], 0, 1).to("cuda")
        for i in range(x.shape[1]):
            pre_out_i = self.OneStep_forward(x[:, i:i + 1, :], pre_out_i)
            custom_output = torch.cat((custom_output, pre_out_i), dim=1)

        condition_lo, condition_up, condition_mid = 0, 0, 0
        return custom_output, self.weights, condition_lo, condition_up, condition_mid

    def forward_test(self, x, _out_prev):
        m_lo, m_up, c_lo, c_up = self.weights[0], self.weights[1], self.weights[2], self.weights[3]
        # 10.4949016, 10.69768651, 0.47326037, -0.16193161
        # y = (m/15) * (x + c/6+55/m/6-0.5)
        # (10.4949016/15) = 0.6996601066666667 , (0.47326037/6+55/10.4949016/6-0.5 = 0.45231671053065725
        # (10.69768651/15) =  0.7131791006666667, -0.16193161/6+55/10.69768651/6-0.5 = 0.329894487784294
        # m_lo, m_up, c_lo, c_up = torch.tensor(0.70),  torch.tensor(0.71),  torch.tensor(0.45),  torch.tensor(0.33)
        f1 = self.h((m_lo.detach() * x + m_lo.detach() * c_lo.detach() - _out_prev) / m_lo.detach())
        f2 = self.h((_out_prev - m_up.detach() * x - m_up.detach() * c_up.detach()) / m_up.detach())
        out_curr = m_lo * x * f1 + m_lo * c_lo * f1 + m_up * x * f2 + m_up * c_up * f2 + _out_prev * (1 - f1) * (1 - f2)
        condition_lo, condition_up, condition_mid = 0, 0, 0
        return out_curr, self.weights, condition_lo, condition_up, condition_mid

    def forward_v1(self, x, _out_prev):
        # Expand dynamic weights for broadcasting
        m_lo, m_up, c_lo, c_up = self.weights[0], self.weights[1], self.weights[2], self.weights[3]

        # Calculate z_lo and z_up
        z_lo = _out_prev / m_lo - c_lo
        z_up = _out_prev / m_up - c_up

        # Initialize output tensor
        out_curr = torch.zeros_like(x)

        # Condition masks
        condition_lo = (x <= z_lo)
        condition_up = (x >= z_up)
        condition_mid = (~condition_lo & ~condition_up)

        # Apply piecewise function using conditions
        out_curr_lo = m_lo * (x + c_lo)
        out_curr_up = m_up * (x + c_up)

        out_curr = torch.where(condition_lo, out_curr_lo, out_curr)
        out_curr = torch.where(condition_up, out_curr_up, out_curr)
        out_curr = torch.where(condition_mid, _out_prev, out_curr)
        # out_curr[condition_lo] = out_curr_lo[condition_lo]
        # out_curr[condition_up] = out_curr_up[condition_up]
        # out_curr[condition_mid] = _out_prev[condition_mid]

        return out_curr, self.weights, condition_lo, condition_up, condition_mid


class fixedBacklash(nn.Module):
    def __init__(self):
        super(fixedBacklash, self).__init__()

    def h(self, s):
        s_copy = s.clone()
        s_copy[s > 0] = 0
        s_copy[s <= 0] = 1
        return s_copy

    def forward(self, x, _out_prev):
        # m_lo, m_up, c_lo, c_up = self.weights[0], self.weights[1], self.weights[2], self.weights[3]
        # 10.4949016, 10.69768651, 0.47326037, 0.16193161
        # a1, a2, b1, b2 = 10.40340795, 10.56855929, 0.50669744, 0.24372073
        # y = (m/15) * (x + c/6+55/m/6-0.5)
        # (a1/15) = 0.6996601066666667 , (b1/6+55/a1/6-0.5 = 0.45231671053065725
        # (a2/15) =  0.7131791006666667, -b2/6+55/a2/6-0.5 = 0.329894487784294
        m_lo, m_up, c_lo, c_up = torch.tensor(0.6847), torch.tensor(0.7132), torch.tensor(0.4703), torch.tensor(0.3338) #0.6847, 0.7132, 0.4703, 0.3338
        # m_lo, m_up, c_lo, c_up = torch.tensor(0.70), torch.tensor(0.71), torch.tensor(0.45), torch.tensor(0.33)
        f1 = self.h((m_lo.detach() * x + m_lo.detach() * c_lo.detach() - _out_prev) / m_lo.detach())
        f2 = self.h((_out_prev - m_up.detach() * x - m_up.detach() * c_up.detach()) / m_up.detach())
        out_curr = m_lo * x * f1 + m_lo * c_lo * f1 + m_up * x * f2 + m_up * c_up * f2 + _out_prev * (1 - f1) * (1 - f2)
        return out_curr


class LSTM_backlash_sum_Net(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(LSTM_backlash_sum_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        self.custom_layer = fixedBacklash()

    def forward(self, points, pre_points, hidden):
        # points: current input
        # pre_points: previous output
        custom_output = self.custom_layer(points, pre_points)

        lstm_out, h_ = self.rnn(points, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)
        output = linear_out + custom_output

        return output, h_, custom_output, linear_out


class LSTM_backlash_sum2_Net(nn.Module):
    # Change the data flow.
    # During training, transmit previous backlash output to the next step,
    # instead of using ground truth or lstm output

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(LSTM_backlash_sum2_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        self.custom_layer = fixedBacklash()

    def forward(self, points, pre_points, hidden):
        # points: current input
        # pre_points: previous output
        lstm_out, h_ = self.rnn(points, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)

        pre_out_i = pre_points[:, 0:1, :]
        custom_output = torch.empty(points.shape[0], 0, 1).to("cuda")
        for i in range(points.shape[1]):
            pre_out_i = self.custom_layer(points[:, i:i + 1, :], pre_out_i)
            custom_output = torch.cat((custom_output, pre_out_i), dim=1)
        # custom_output = self.custom_layer(points, pre_points)
        output = linear_out + custom_output

        return output, h_, custom_output, linear_out


class LSTM_backlash_sum3_Net(nn.Module):
    # Change the data flow.
    # During training, transmit previous backlash output to the next step,
    # instead of using ground truth or lstm output

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(LSTM_backlash_sum3_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, points, backlash_out, hidden):
        # points: current input
        # pre_points: previous output
        lstm_out, h_ = self.rnn(points, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)

        output = linear_out + backlash_out

        return output, h_, linear_out


class backlash_LSTM_Net(nn.Module):
    # Change the data flow.
    # During training, transmit previous backlash output to the next step,
    # instead of using ground truth or lstm output

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(backlash_LSTM_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, points, backlash_out, hidden):
        # points: current input
        # pre_points: previous output
        lstm_out, h_ = self.rnn(backlash_out, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)

        return linear_out, h_, linear_out

class fixedBacklashInverseNet(nn.Module):
    def __init__(self):
        super(fixedBacklashInverseNet, self).__init__()
        self.weights = nn.Parameter(torch.randn(4))

    def h(self, s):
        """Step function, see (4) in [1]"""
        s_copy = s.clone()
        s_copy[s > 0] = 1
        s_copy[s <= 0] = 0
        return s_copy

    def forward(self, x_curr, _x_prev, u_pre):
        # m_lo, m_up, c_lo, c_up = self.weights[0], self.weights[1], self.weights[2], self.weights[3]  #0.6931, 0.7032, 0.4523, 0.3573
        m_lo, m_up, c_lo, c_up = torch.tensor(0.6931), torch.tensor(0.7032), torch.tensor(0.4523), torch.tensor(0.3573) # 0.6931, 0.7032, 0.4523, 0.3573
        f1 = self.h(x_curr - _x_prev)
        f2 = self.h(_x_prev - x_curr)
        u = (x_curr / m_up - c_up) * f1 + (x_curr / m_lo - c_lo) * f2 + u_pre * (1 - f1) * (1 - f2)
        return u

class BacklashInverseNet(nn.Module):
    def __init__(self):
        super(BacklashInverseNet, self).__init__()
        self.weights = nn.Parameter(torch.randn(4))

    def h(self, s):
        """Step function, see (4) in [1]"""
        s_copy = s.clone()
        s_copy[s > 0] = 1
        s_copy[s <= 0] = 0
        return s_copy

    def forward(self, x_curr, _x_prev, u_pre):
        m_lo, m_up, c_lo, c_up = self.weights[0], self.weights[1], self.weights[2], self.weights[3]
        # m_lo, m_up, c_lo, c_up = torch.tensor(0.694), torch.tensor(0.705), torch.tensor(0.466), torch.tensor(0.327)
        f1 = self.h(x_curr - _x_prev)
        f2 = self.h(_x_prev - x_curr)
        u = (x_curr / m_up - c_up) * f1 + (x_curr / m_lo - c_lo) * f2 + u_pre * (1 - f1) * (1 - f2)
        return u, self.weights

    # def forward(self, x_curr, _x_prev, u_pre):
    #     u_pre_i = u_pre[:, 0:1, :]
    #     custom_output = torch.empty(x_curr.shape[0], 0, 1).to("cuda")
    #     for i in range(x_curr.shape[1]):
    #         u_pre_i = self.OneStep_forward(x_curr[:, i:i + 1, :], _x_prev[:, i:i + 1, :], u_pre_i)
    #         custom_output = torch.cat((custom_output, u_pre_i), dim=1)
    #     return custom_output, self.weights


class BacklashInv(nn.Module):

    def __init__(self):
        super(BacklashInv, self).__init__()

    def h(self, s):
        """Step function, see (4) in [1]"""
        s_copy = s.clone()
        s_copy[s > 0] = 1
        s_copy[s <= 0] = 0
        return s_copy

    def forward(self, x_curr, _x_prev, u_pre, weights):
        m_lo, m_up, c_lo, c_up = (weights[:, :, 0].unsqueeze(-1), weights[:, :, 1].unsqueeze(-1),
                                  weights[:, :, 2].unsqueeze(-1), weights[:, :, 3].unsqueeze(-1))
        # m_lo, m_up, c_lo, c_up = torch.tensor(0.694), torch.tensor(0.705), torch.tensor(0.466), torch.tensor(0.327)
        f1 = self.h(x_curr - _x_prev)
        f2 = self.h(_x_prev - x_curr)
        u = (x_curr / m_up - c_up) * f1 + (x_curr / m_lo - c_lo) * f2 + u_pre * (1 - f1) * (1 - f2)
        return u, weights


class LSTMBacklashInverseNet(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(LSTMBacklashInverseNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 4  # for 4 parameters in backlash model
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        self.custom_layer = BacklashInv()

    def forward(self, x_curr, _x_prev, u_pre, hidden):
        # points: current input
        # pre_points: previous output
        lstm_out, h_ = self.rnn(x_curr, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)

        custom_output, dynamic_weights = self.custom_layer(x_curr, _x_prev, u_pre, linear_out)
        return custom_output, h_, dynamic_weights


class LSTMBacklashSumInvNet(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, num_layers=2, dropout=0, act=None):
        super(LSTMBacklashSumInvNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1  # for 4 parameters in backlash model
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x_curr, backlash_out, hidden):
        # points: current input
        # pre_points: previous output
        lstm_out, h_ = self.rnn(x_curr, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)

        output = linear_out + backlash_out
        return output, h_, linear_out

class fixedBacklashTipY(nn.Module):
    def __init__(self, datatype="stair"):
        super(fixedBacklashTipY, self).__init__()
        if datatype == "stair":
            self.p = [0.2946, 0.2805, 0.7793, 0.7425]
        else:
            self.p = [0.2891, 0.2655, 0.7899, 0.7924]

    def h(self, s):
        s_copy = s.clone()
        s_copy[s > 0] = 0
        s_copy[s <= 0] = 1
        return s_copy

    def forward(self, x, _out_prev):
        m_lo, m_up, c_lo, c_up = torch.tensor(self.p[0]), torch.tensor(self.p[1]), torch.tensor(self.p[2]), torch.tensor(self.p[3]) #[0.2946, 0.2805, 0.7793, 0.7425]
        f1 = self.h((m_lo.detach() * x + m_lo.detach() * c_lo.detach() - _out_prev) / m_lo.detach())
        f2 = self.h((_out_prev - m_up.detach() * x - m_up.detach() * c_up.detach()) / m_up.detach())
        out_curr = m_lo * x * f1 + m_lo * c_lo * f1 + m_up * x * f2 + m_up * c_up * f2 + _out_prev * (1 - f1) * (1 - f2)
        return out_curr

class fixedBacklashTipYInverseNet(nn.Module):
    def __init__(self, datatype="stair"):
        super(fixedBacklashTipYInverseNet, self).__init__()
        if datatype == "stair":
            self.p = [0.2920, 0.2924, 0.7879, 0.6935]
        else:
            self.p = [0.2804, 0.2778, 0.8205, 0.7422]

    def h(self, s):
        """Step function, see (4) in [1]"""
        s_copy = s.clone()
        s_copy[s > 0] = 1
        s_copy[s <= 0] = 0
        return s_copy

    def forward(self, x_curr, _x_prev, u_pre):
        m_lo, m_up, c_lo, c_up = torch.tensor(self.p[0]), torch.tensor(self.p[1]), torch.tensor(self.p[2]), torch.tensor(self.p[3]) #[0.2920, 0.2924, 0.7879, 0.6935]
        f1 = self.h(x_curr - _x_prev)
        f2 = self.h(_x_prev - x_curr)
        u = (x_curr / m_up - c_up) * f1 + (x_curr / m_lo - c_lo) * f2 + u_pre * (1 - f1) * (1 - f2)
        return u