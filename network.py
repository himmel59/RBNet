
import torch
import torch.nn as nn

MUL = 5
THRESHOLD1 = 1
THRESHOLD2 = 1
EPSILON = 1e-36


class MLP1(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=128, out_dim=50, eta=1e-4):
        torch.set_default_dtype(torch.float64)
        super(MLP1, self).__init__()

        self.eta = eta

        self.pre00 = nn.Tanh()
        self.pre01 = nn.Linear(in_dim, hidden_dim)
        self.pre02 = nn.LayerNorm(hidden_dim)
        self.pre03 = nn.ReLU(True)

        self.main01 = nn.Linear(hidden_dim, hidden_dim)
        self.main02 = nn.LayerNorm(hidden_dim)
        self.main03 = nn.ReLU(True)
        self.main11 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.main12 = nn.LayerNorm(hidden_dim * 2)
        self.main13 = nn.ReLU(True)
        self.main21 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.main22 = nn.LayerNorm(hidden_dim * 2)
        self.main23 = nn.ReLU(True)
        self.main31 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.main32 = nn.LayerNorm(hidden_dim * 2)
        self.main33 = nn.ReLU(True)

        self.postv1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.postv2 = nn.LayerNorm(hidden_dim)
        self.postv3 = nn.ReLU(True)
        self.postv4 = nn.Linear(hidden_dim, out_dim)
        self.postv5 = nn.LayerNorm(out_dim)
        self.postv6 = nn.ReLU(True)
        self.postv7 = nn.Linear(out_dim, 1)

        self.postx1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.postx2 = nn.LayerNorm(hidden_dim)
        self.postx3 = nn.ReLU(True)
        self.postx4 = nn.Linear(hidden_dim, out_dim)
        self.postx5 = nn.LayerNorm(out_dim)
        self.postx6 = nn.ReLU(True)
        self.postx7 = nn.Linear(out_dim, 1)
        self.postx8 = nn.Tanh()

    def forward(self, x):
        torch.set_default_dtype(torch.float64)
        exc = x.transpose(0, 1)[15:16].transpose(0, 1)

        x = torch.log(torch.abs(x) + self.eta)
        x = self.pre00(x)
        x = self.pre01(x)  # 16 -> 128
        x = self.pre02(x)
        x = self.pre03(x)
        x = self.main01(x)  # 128 -> 128
        x = self.main02(x)
        x = self.main03(x)
        x = self.main11(x)  # 128 -> 256
        x = self.main12(x)
        x = self.main13(x)
        x = self.main21(x)  # 256 -> 256
        x = self.main22(x)
        x = self.main23(x)
        x = self.main31(x)  # 256 -> 256
        x = self.main32(x)
        x = self.main33(x)

        v = self.postv1(x)  # 256 -> 128
        v = self.postv2(v)
        v = self.postv3(v)
        v = self.postv4(v)  # 128 -> 50
        v = self.postv5(v)
        v = self.postv6(v)
        v = self.postv7(v)  # 50 -> 1

        x = self.postx1(x)  # 256 -> 128
        x = self.postx2(x)
        x = self.postx3(x)
        x = self.postx4(x)  # 128 -> 50
        x = self.postx5(x)
        x = self.postx6(x)
        x = self.postx7(x)  # 50 -> 1
        x = self.postx8(x)

        # x is correction percentage at first
        # correction should not bigger than THRESHOLD1 * exc
        # x should be unweighted exc potential
        x = x * THRESHOLD1
        x = x * exc  # exc is unweighted exc potential

        # v should be 2log(sigma)
        # sigma/x should not bigger than THRESHOLD2
        # v should not bigger than log(THRESHOLD^2)+log(x^2)

        v = torch.clamp(v, max=torch.log((THRESHOLD2 * x)**2 + EPSILON))

        # x is the mu of normal distribution, v is the sigma of normal distribution
        # x is not includes original exc
        return x, v


class MLP1A(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=128, out_dim=50, eta=1e-4):
        torch.set_default_dtype(torch.float64)
        super(MLP1A, self).__init__()

        self.eta = eta

        self.pre00 = nn.Tanh()
        self.pre01 = nn.Linear(in_dim, hidden_dim)
        self.pre02 = nn.LayerNorm(hidden_dim)
        self.pre03 = nn.ReLU(True)

        self.main01 = nn.Linear(hidden_dim, hidden_dim)
        self.main02 = nn.LayerNorm(hidden_dim)
        self.main03 = nn.ReLU(True)
        self.main11 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.main12 = nn.LayerNorm(hidden_dim * 2)
        self.main13 = nn.ReLU(True)
        self.main21 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.main22 = nn.LayerNorm(hidden_dim * 2)
        self.main23 = nn.ReLU(True)
        self.main31 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.main32 = nn.LayerNorm(hidden_dim * 2)
        self.main33 = nn.ReLU(True)

        self.postv1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.postv2 = nn.LayerNorm(hidden_dim)
        self.postv3 = nn.ReLU(True)
        self.postv4 = nn.Linear(hidden_dim, out_dim)
        self.postv5 = nn.LayerNorm(out_dim)
        self.postv6 = nn.ReLU(True)
        self.postv7 = nn.Linear(out_dim, 1)

        self.postx1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.postx2 = nn.LayerNorm(hidden_dim)
        self.postx3 = nn.ReLU(True)
        self.postx4 = nn.Linear(hidden_dim, out_dim)
        self.postx5 = nn.LayerNorm(out_dim)
        self.postx6 = nn.ReLU(True)
        self.postx7 = nn.Linear(out_dim, 1)
        self.postx8 = nn.Tanh()

    def forward(self, x):
        torch.set_default_dtype(torch.float64)
        exc = x.transpose(0, 1)[15:16].transpose(0, 1)

        x = torch.log(torch.abs(x) + self.eta)
        x = self.pre00(x)
        x = self.pre01(x)  # 16 -> 128
        x = self.pre02(x)
        x = self.pre03(x)
        x = self.main01(x)  # 128 -> 128
        x = self.main02(x)
        x = self.main03(x)
        x = self.main11(x)  # 128 -> 256
        x = self.main12(x)
        x = self.main13(x)
        x = self.main21(x)  # 256 -> 256
        x = self.main22(x)
        x = self.main23(x)
        x = self.main31(x)  # 256 -> 256
        x = self.main32(x)
        x = self.main33(x)

        # v = self.postv1(x)  # 256 -> 128
        # v = self.postv2(v)
        # v = self.postv3(v)
        # v = self.postv4(v)  # 128 -> 50
        # v = self.postv5(v)
        # v = self.postv6(v)
        # v = self.postv7(v)  # 50 -> 1

        x = self.postx1(x)  # 256 -> 128
        x = self.postx2(x)
        x = self.postx3(x)
        x = self.postx4(x)  # 128 -> 50
        x = self.postx5(x)
        x = self.postx6(x)
        x = self.postx7(x)  # 50 -> 1
        x = self.postx8(x)

        v = torch.zeros_like(x).cuda()
        # x is correction percentage at first
        # correction should not bigger than THRESHOLD1 * exc
        # x should be unweighted exc potential
        x = x * THRESHOLD1
        x = x * exc  # exc is unweighted exc potential

        # v should be 2log(sigma)
        # sigma/x should not bigger than THRESHOLD2
        # v should not bigger than log(THRESHOLD^2)+log(x^2)

        # v = torch.clamp(v, max=torch.log((THRESHOLD2 * x)**2 + EPSILON))

        # x is the mu of normal distribution, v is the sigma of normal distribution
        # x is not includes original exc
        return x, v





class MLP1B(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=128, out_dim=50, eta=1e-4):
        torch.set_default_dtype(torch.float64)
        super(MLP1B, self).__init__()

        self.eta = eta

        self.pre00 = nn.Tanh()
        self.pre01 = nn.Linear(in_dim, hidden_dim)
        self.pre02 = nn.LayerNorm(hidden_dim)
        self.pre03 = nn.ReLU(True)

        self.main01 = nn.Linear(hidden_dim, hidden_dim)
        self.main02 = nn.LayerNorm(hidden_dim)
        self.main03 = nn.ReLU(True)
        self.main11 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.main12 = nn.LayerNorm(hidden_dim * 2)
        self.main13 = nn.ReLU(True)
        self.main21 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.main22 = nn.LayerNorm(hidden_dim * 2)
        self.main23 = nn.ReLU(True)
        self.main31 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.main32 = nn.LayerNorm(hidden_dim * 2)
        self.main33 = nn.ReLU(True)

        self.postv1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.postv2 = nn.LayerNorm(hidden_dim)
        self.postv3 = nn.ReLU(True)
        self.postv4 = nn.Linear(hidden_dim, out_dim)
        self.postv5 = nn.LayerNorm(out_dim)
        self.postv6 = nn.ReLU(True)
        self.postv7 = nn.Linear(out_dim, 1)

        self.postx1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.postx2 = nn.LayerNorm(hidden_dim)
        self.postx3 = nn.ReLU(True)
        self.postx4 = nn.Linear(hidden_dim, out_dim)
        self.postx5 = nn.LayerNorm(out_dim)
        self.postx6 = nn.ReLU(True)
        self.postx7 = nn.Linear(out_dim, 1)
        self.postx8 = nn.Tanh()

    def forward(self, x):
        torch.set_default_dtype(torch.float64)
        exc = x.transpose(0, 1)[15:16].transpose(0, 1)

        x = torch.log(torch.abs(x) + self.eta)
        x = self.pre00(x)
        x = self.pre01(x)  # 16 -> 128
        x = self.pre02(x)
        x = self.pre03(x)
        x = self.main01(x)  # 128 -> 128
        x = self.main02(x)
        x = self.main03(x)
        x = self.main11(x)  # 128 -> 256
        x = self.main12(x)
        x = self.main13(x)
        x = self.main21(x)  # 256 -> 256
        x = self.main22(x)
        x = self.main23(x)
        x = self.main31(x)  # 256 -> 256
        x = self.main32(x)
        x = self.main33(x)

        # v = self.postv1(x)  # 256 -> 128
        # v = self.postv2(v)
        # v = self.postv3(v)
        # v = self.postv4(v)  # 128 -> 50
        # v = self.postv5(v)
        # v = self.postv6(v)
        # v = self.postv7(v)  # 50 -> 1

        x = self.postx1(x)  # 256 -> 128
        x = self.postx2(x)
        x = self.postx3(x)
        x = self.postx4(x)  # 128 -> 50
        x = self.postx5(x)
        x = self.postx6(x)
        x = self.postx7(x)  # 50 -> 1
        x = self.postx8(x)

        # x is correction percentage at first
        # correction should not bigger than THRESHOLD1 * exc
        # x should be unweighted exc potential
        x = x * THRESHOLD1
        x = x * exc  # exc is unweighted exc potential
        v = torch.zeros_like(x).cuda()
        # v should be 2log(sigma)
        # sigma/x should not bigger than THRESHOLD2
        # v should not bigger than log(THRESHOLD^2)+log(x^2)

        # v = torch.clamp(v, max=torch.log((THRESHOLD2 * x)**2 + EPSILON))

        # x is the mu of normal distribution, v is the sigma of normal distribution
        # x is not includes original exc
        return x, v

