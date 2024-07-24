import torch
import torch.nn as nn
import torch.nn.functional as F



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

class GRDPINet(nn.Module):
    def __init__(self):
        super(GRDPINet, self).__init__()
        self.weights = nn.Parameter(torch.ones(11)*0.5)
        # self.weights = nn.Parameter(torch.tensor([0.5, 0.35, 0.25, 0.2, 0.15, 2, 0.5, 0.28, 50, 1, 0.01]))
        # self.weights = nn.Parameter(torch.randn(11))
        self.tanh = nn.Tanh()


    def zi(self, u, r, z_pre):
        # z = torch.max(u-r, torch.min(u+r, z_pre))
        z = (u - r) + torch.relu(((u + r) - torch.relu((u + r) - z_pre)) - (u - r))
        # leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        return z

    def ri(self, h, i, beta, du):
        # return h*i + beta*torch.abs(du)
        return h*i + beta*torch.sqrt(du*du + 1e-10)

    def OneStep_forward(self, u, u_pre, z_pre):
        # self.weights.data.clamp_(min=1e-10)  # Ensure w > 0
        # self.weights[6].data.clamp_(max=self.weights[5].data.item() * 0.5)  # Ensure w1 >> w2
        a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
                                                      , self.weights[5], self.weights[6], self.weights[7], self.weights[8], self.weights[9], self.weights[10])
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 1.0263, 0.5861, 0.5499, 0.5201, 0.5005, 0.0064, 1.1270, 0.2824, 1.1937, 0.3281, 0.5558
        # a0, a1, a2, a3, a4, h, beta_ratio, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
        #                                               , self.weights[5], self.weights[6], self.weights[7], self.weights[8], self.weights[9], self.weights[10])
        # beta = h/(10+beta_ratio)
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.5, 0.35, 0.25, 0.2, 0.15, 2, 1, 0.28, 50, 1, 0.01
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.6838, 0.8846, 0.9016, 0.8162, 0.6794, 0.0032*1000, 1.0168/100, 0.5000, 1.3850, 0.1705, 0.7050
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.0402, 1.2204, 1.2358, 1.0049, 0.6633, 0.0028, 0.8185, 0.5000, 1.7420, 0.1498, 0.9134
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 3.9502e-01, 3.4013e-01, 2.0201e-01, 1.1240e-01, 4.8256e-02, 1.5750e+00, \
        #                                              7.8748e-01, 2.9991e-01, 4.9959e+01, 1.0000e+00, 1.2204e-02

        # u = c1*self.tanh(c2*u)
        # u_pre = c1*self.tanh(c2*u_pre)
        # u = c1 * u
        # u_pre = c1 * u_pre
        # h = torch.clamp(self.weights[5], 0, 100)
        # beta = torch.clamp(self.weights[6], 0, 100)
        # a0, a1, a2, a3, a4 = self.weights[0], self.weights[1], 0, 0, 0
        # h = 0.02
        # beta = 1
        # a0, a1, a2, a3, a4, h, beta = 0.3124, 0.1906, 0.0461, 0.1, 0.1, 0.03, 1
        du = (u - u_pre)
        r1 = self.ri(h, 1, beta, du)
        r2 = self.ri(h, 2, beta, du)
        r3 = self.ri(h, 3, beta, du)
        r4 = self.ri(h, 4, beta, du)
        z1, z2 = self.zi(u, r1, z_pre[:, :, 0:1]), self.zi(u, r2, z_pre[:, :, 1:2])
        z3, z4 = self.zi(u, r3, z_pre[:, :, 2:3]), self.zi(u, r4, z_pre[:, :, 3:4])
        out_curr = a0*u + a1*z1 + a2*z2 + a3*z3 + a4*z4 + b
        z = torch.cat([z1, z2, z3, z4], dim=2)
        # out_curr = c3 * out_curr + b
        return out_curr, z

    def OneStep_forward_(self, u, u_pre, z_pre):
        self.weights.data.clamp_(min=1e-8)  # Ensure w > 0
        # self.weights[6].data.clamp_(max=self.weights[5].data.item() * 0.5)  # Ensure w1 >> w2
        a0, a1, h, beta, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
                                                      , self.weights[5], self.weights[6], self.weights[7])

        u = c1*self.tanh(c2*u)
        u_pre = c1*self.tanh(c2*u_pre)
        du = (u - u_pre)
        r1 = self.ri(h, 1, beta, du)
        z1 = self.zi(u, r1, z_pre[:, :, 0:1])
        out_curr = a0*u + a1*z1
        z = torch.cat([z1], dim=2)
        out_curr = c3 * out_curr + b
        return out_curr, z

    def forward(self, x, x_pre, z_init):
        z = z_init  # shape: (batch_size, 1, 4)
        custom_output = torch.empty(x.shape[0], 0, 1).to("cuda")
        for i in range(x.shape[1]):
            out_i, z = self.OneStep_forward(x[:, i:i + 1, :], x_pre[:, i:i + 1, :], z)
            custom_output = torch.cat((custom_output, out_i), dim=1)

        return custom_output, self.weights, z


class GRDPI_inv_Net(nn.Module):
    def __init__(self):
        super(GRDPI_inv_Net, self).__init__()
        self.weights = nn.Parameter(torch.ones(11)*0.5)
        # self.weights = nn.Parameter(torch.randn(8))


    def zi(self, u, r, z_pre):
        # z = torch.max(u-r, torch.min(u+r, z_pre))
        z = (u - r) + torch.relu(((u + r) - torch.relu((u + r) - z_pre)) - (u - r))
        # leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # z = (u - r) + F.leaky_relu(((u + r) - F.leaky_relu((u + r) - z_pre, negative_slope=0.01)) - (u - r),
        #                            negative_slope=0.01)
        return z

    def ri(self, h, i, beta, du):
        # return h*i + beta*torch.abs(du)
        return h*i + beta*torch.sqrt(du*du + 0.0000000001)

    def OneStep_forward_(self, u, u_pre, z_pre):
        # self.weights.data.clamp_(min=1e-8)  # Ensure w > 0
        a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
                                                      , self.weights[5], self.weights[6], self.weights[7], self.weights[8], self.weights[9], self.weights[10])

        # u = (u - b) / c3
        # u_pre = (u_pre - b) / c3
        du = (u - u_pre)
        r1 = self.ri(h, 1, beta, du)
        r2 = self.ri(h, 2, beta, du)
        r3 = self.ri(h, 3, beta, du)
        r4 = self.ri(h, 4, beta, du)
        z1, z2 = self.zi(u, r1, z_pre[:, :, 0:1]), self.zi(u, r2, z_pre[:, :, 1:2])
        z3, z4 = self.zi(u, r3, z_pre[:, :, 2:3]), self.zi(u, r4, z_pre[:, :, 3:4])
        # out_curr = a0 * u + a1 * z1 + a2 * z2 + a3 * z3 + a4 * z4 + b
        out_curr = a0 * u + a1 * z1 + b
        z = torch.cat([z1, z2, z3, z4], dim=2)
        # print("r1 shape:", r1.shape, "z shape:", z.shape)
        # out_curr = torch.atanh(out_curr/c1)/c2
        return out_curr, z

    def OneStep_forward__(self, u, u_pre, z_pre):
        a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
                                                        , self.weights[5], self.weights[6], self.weights[7], self.weights[8], self.weights[9], self.weights[10])
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.2961, 1.1155, 1.0423, 0.8065, 0.5944, 0.0038, 0.8095, 0.2850, 1.4443, 0.1689, 0.7272
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.6838, 0.8846, 0.9016, 0.8162, 0.6794, 0.0032*1000, 1.0168/100, 0.5000, 1.3850, 0.1705, 0.7050
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.0402, 1.2204, 1.2358, 1.0049, 0.6633, 0.0028, 0.8185, 0.5000, 1.7420, 0.1498, 0.9134
        # a0, a1, a2, a3, a4, h, beta, c1, c2, c3 = 0.5, 0.3667, 0.2690, 0.1973, 0.1447, 3, 28 / 400, 50, 1 / 40, 0.9
        # a0, a1, a2, a3, a4, h, beta_ratio, b, c1, c2, c3 = 1.0495, 0.5077, 0.4721, 0.4497, 0.4390, 0.0103, 0.3676, 0.2783, 1.1553, 0.4109, 0.5131
        # beta = h/(100+beta_ratio)
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.5, 0.35, 0.25, 0.2, 0.15, 2, 1, 0.28, 50, 1, 0.01
        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 3.9502e-01, 3.4013e-01, 2.0201e-01, 1.1240e-01, 4.8256e-02, 1.5750e+00,\
        #                                              7.8748e-01, 2.9991e-01, 4.9959e+01, 1.0000e+00, 1.2204e-02

        # a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.2962, 1.1139, 1.0392, 0.8034, 0.5933, 0.0032, 0.8053, 0.2873, 1.4483, 0.1680, 0.7274  #  Sinus data for training with gamma
        a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = 0.1422, 0.1403, 0.1361, 0.1341, 0.1339, 0.0222, 0.6136, 0.2908, 0.5000, 0.5000, 0.5000  #  Sinus data for training no gamma

        # u = (u-b)/c3
        # u_pre = (u_pre-b)/c3
        u = (u-b)
        u_pre = (u_pre-b)
        du = (u - u_pre)
        r1 = self.ri(h, 1, beta, du)
        r2 = self.ri(h, 2, beta, du)
        r3 = self.ri(h, 3, beta, du)
        r4 = self.ri(h, 4, beta, du)
        s1 = a0*r1
        s2 = s1 + (a0+a1)*(r2-r1)
        s3 = s2 + (a0+a1+a2)*(r3-r2)
        s4 = s3 + (a0+a1+a2+a3)*(r4-r3)
        z1, z2 = self.zi(u, s1, z_pre[:, :, 0:1]), self.zi(u, s2, z_pre[:, :, 1:2])
        z3, z4 = self.zi(u, s3, z_pre[:, :, 2:3]), self.zi(u, s4, z_pre[:, :, 3:4])
        b0 = 1/a0
        b1 = 1/(a0+a1) - 1/(a0)
        b2 = 1/(a0+a1+a2) - 1/(a0+a1)
        b3 = 1/(a0+a1+a2+a3) - 1/(a0+a1+a2)
        b4 = 1/(a0+a1+a2+a3+a4) - 1/(a0+a1+a2+a3)
        out_curr = b0 * u + b1 * z1 + b2 * z2 + b3 * z3 + b4 * z4
        z = torch.cat([z1, z2, z3, z4], dim=2)
        # print("r1 shape:", r1.shape, "z shape:", z.shape)
        # out_curr = torch.atanh(out_curr/c1)/c2
        # out_curr = out_curr/c1
        return out_curr, z

    def OneStep_forward(self, u, u_pre, z_pre):
        epsilon = 1e-10
        a0, a1, a2, a3, a4, h, beta, b, c1, c2, c3 = (self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4]
                                                      , self.weights[5], self.weights[6], self.weights[7], self.weights[8], self.weights[9], self.weights[10])

        # a0, a1, h, beta, b, c1, c2, c3 =  0.7463, 0.5401, 0.0854, 2.1053, 0.2819, 2.0407, 0.4986, 0.5523 #  Sinus data for training

        # u = (u-b)/(c3+epsilon)
        # u_pre = (u_pre-b)/(c3+epsilon)
        du = (u - u_pre)
        s1 = a0*(h + beta*torch.sqrt(du*du + epsilon))
        s2 = (2*a0+a1)*h + a0*beta*torch.sqrt(du*du + epsilon)
        s3 = (3*a0+2*a1+a2)*h + a0*beta*torch.sqrt(du*du + epsilon)
        s4 = (4*a0+3*a1+2*a2+a3)*h + a0*beta*torch.sqrt(du*du + epsilon)
        z1, z2 = self.zi(u, s1, z_pre[:, :, 0:1]), self.zi(u, s2, z_pre[:, :, 1:2])
        z3, z4 = self.zi(u, s3, z_pre[:, :, 2:3]), self.zi(u, s4, z_pre[:, :, 3:4])
        b0 = 1/(a0+epsilon)
        b1 = 1/(a0+a1+epsilon) - 1/(a0+epsilon)
        b2 = 1 / (a0 + a1 + a2+epsilon) - 1 / (a0 + a1+epsilon)
        b3 = 1 / (a0 + a1 + a2 + a3+epsilon) - 1 / (a0 + a1 + a2+epsilon)
        b4 = 1 / (a0 + a1 + a2 + a3 + a4+epsilon) - 1 / (a0 + a1 + a2 + a3+epsilon)
        # s1 = r1
        # z1 = self.zi(u, s1, z_pre[:, :, 0:1])
        # b0 = a0
        # b1 = a1
        out_curr = b0 * u + b1 * z1 + b2 * z2 + b3 * z3 + b4 * z4 + b
        z = torch.cat([z1, z2, z3, z4], dim=2)
        # print("r1 shape:", r1.shape, "z shape:", z.shape)
        # out_curr = torch.clamp(out_curr, min=-1 + epsilon, max=1 - epsilon)
        # out_curr = torch.atanh(out_curr/(c1+epsilon))/(c2+epsilon)
        # out_curr = out_curr/c1
        return out_curr, z


    def forward(self, x, x_pre, z_init):
        z = z_init  # shape: (batch_size, 1, 4)
        custom_output = torch.empty(x.shape[0], 0, 1).to("cuda")
        for i in range(x.shape[1]):
            out_i, z = self.OneStep_forward(x[:, i:i + 1, :], x_pre[:, i:i + 1, :], z)
            custom_output = torch.cat((custom_output, out_i), dim=1)

        return custom_output, self.weights, z