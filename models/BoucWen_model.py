import torch
import torch.nn as nn
import torch.nn.functional as F

class BoucWenNet(nn.Module):
    def __init__(self):
        super(BoucWenNet, self).__init__()
        self.weights = nn.Parameter(torch.randn(6))

    def OneStep_forward(self, x, x_pre, h_pre):
        # alpha1, alpha2, A, v, delta, a, b = self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4], self.weights[5], self.weights[6]
        alpha1, alpha2, A, v, delta, b = self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4], self.weights[5]
        # alpha1 = 2.23
        # alpha2 = 0.62
        # delta = 1
        dx = x - x_pre
        alpha_x = (alpha1*torch.exp(2*dx)+alpha2)/(torch.exp(2*dx) + 1)
        h = (A*dx+delta*torch.abs(dx)+h_pre) / (1+v*torch.abs(dx))
        out_curr = alpha_x*x + h + b
        return out_curr, h

    def forward(self, x, x_pre, h_init):
        h = h_init  # shape: (batch_size, 1, 1)
        custom_output = torch.empty(x.shape[0], 0, 1).to("cuda")
        for i in range(x.shape[1]):
            out_i, h = self.OneStep_forward(x[:, i:i + 1, :], x_pre[:, i:i + 1, :], h)
            custom_output = torch.cat((custom_output, out_i), dim=1)

        return custom_output, self.weights, h