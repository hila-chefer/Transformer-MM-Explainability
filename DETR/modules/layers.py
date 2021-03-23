import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye', 'Tanh',  'Mul', 'MatMul', 'WithPosEmbd', 'MultiheadAttention']


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if len(input) == 0:
        return
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R

    def RAP_relprop(self, R_p):
        return R_p


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)[0]
            if torch.is_tensor(self.X) == False:
                Rp = []
                Rp.append(self.X[0] * Cp)
                Rp.append(self.X[1] * Cp)
            else:
                Rp = self.X * (Cp)
            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class WithPosEmbd(RelProp):
    def __init__(self):
        super(WithPosEmbd, self).__init__()
        self.add_pos = Add()

    def forward(self, tensor, pos=None):
        if pos is None:
            self.with_pos = False
            return tensor
        else:
            self.with_pos = True
            return self.add_pos([tensor, pos])

    def relprop(self, cam, alpha, **kwargs):
        return cam
        # if not self.with_pos:
        #     return cam
        # else:
        #     return self.add_pos.relprop(cam, alpha, **kwargs)[0]

class AddEye(RelPropSimple):
    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)

class MatMul(RelPropSimple):
    def forward(self, inputs):
        return torch.matmul(*inputs)

    def relprop(self, R, alpha):
        x1_pos = self.X[0].clamp(min=0)
        x1_neg = self.X[0].clamp(max=0)
        x2_pos = self.X[1].clamp(min=0)
        x2_neg = self.X[1].clamp(max=0)

        Z1 = self.forward([x1_pos, x2_pos])
        Z2 = self.forward([x1_neg, x2_neg])
        S1 = safe_divide(R, Z1)
        S2 = safe_divide(R, Z2)
        C1 = x1_pos * self.gradprop(Z1, x1_pos, S1)[0]
        C2 = x1_neg * self.gradprop(Z2, x1_neg, S2)[0]

        x1_outputs = C1 + C2

        ####
        x1_pos = self.X[0].clamp(min=0)
        x1_neg = self.X[0].clamp(max=0)
        x2_pos = self.X[1].clamp(min=0)
        x2_neg = self.X[1].clamp(max=0)

        Z1 = self.forward([x1_pos, x2_pos])
        Z2 = self.forward([x1_neg, x2_neg])
        S1 = safe_divide(R, Z1)
        S2 = safe_divide(R, Z2)
        C1 = x2_pos * self.gradprop(Z1, x2_pos, S1)[0]
        C2 = x2_neg * self.gradprop(Z2, x2_neg, S2)[0]

        x2_outputs = C1 + C2

        outputs = [x1_outputs / 2, x2_outputs / 2]

        return outputs

class Mul(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)

class ReLU(nn.ReLU, RelProp):
    pass

class Tanh(nn.Tanh, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        # global_min = min(a.min().item(), b.min().item())
        # a = a - global_min
        # b = b - global_min

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs

class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs



class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = []
            for _ in range(self.num):
                Z.append(self.X)

            Spp = []
            Spn = []
            for z, rp, rn in zip(Z, R_p):
                Spp.append(safe_divide(torch.clamp(rp, min=0), z))
                Spn.append(safe_divide(torch.clamp(rp, max=0), z))

            Cpp = self.gradprop(Z, self.X, Spp)[0]
            Cpn = self.gradprop(Z, self.X, Spn)[0]

            Rp = self.X * (Cpp * Cpn)

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X, self.dim)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)

            Rp = []

            for x, cp in zip(self.X, Cp):
                Rp.append(x * (cp))

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R

    def RAP_relprop(self, Rp):
        for m in reversed(self._modules.values()):
            Rp = m.RAP_relprop(Rp)
        return Rp


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R

    def RAP_relprop(self, R_p):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1

        def backward(R_p):
            X = self.X

            weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))

            if torch.is_tensor(self.bias):
                bias = self.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
                                     R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if torch.is_tensor(self.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        # if self.X.min() == self.X.max() == 0:
        #     return R

        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R_out = alpha * activator_relevances - beta * inhibitor_relevances

        R_out = R_out * safe_divide(R.sum(), R_out.sum())

        return R_out

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n

            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=-1, keepdim=True) - R.sum(dim=-1, keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.linear(x1, w1) * R_nonzero
            Za2 = - F.linear(x1, w2) * R_nonzero

            Zb1 = - F.linear(x2, w1) * R_nonzero
            Zb2 = F.linear(x2, w2) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)

            return C1 + C2

        def first_prop(pd, px, nx, pw, nw):
            Rpp = F.linear(px, pw) * pd
            Rpn = F.linear(px, nw) * pd
            Rnp = F.linear(nx, pw) * pd
            Rnn = F.linear(nx, nw) * pd
            Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
            Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

            Z1 = F.linear(px, pw)
            Z2 = F.linear(px, nw)
            Z3 = F.linear(nx, pw)
            Z4 = F.linear(nx, nw)

            S1 = safe_divide(Rpp, Z1)
            S2 = safe_divide(Rpn, Z2)
            S3 = safe_divide(Rnp, Z3)
            S4 = safe_divide(Rnn, Z4)
            C1 = px * self.gradprop(Z1, px, S1)[0]
            C2 = px * self.gradprop(Z2, px, S2)[0]
            C3 = nx * self.gradprop(Z3, nx, S3)[0]
            C4 = nx * self.gradprop(Z4, nx, S4)[0]
            bp = self.bias * pd * safe_divide(Pos, Pos + Neg)
            bn = self.bias * pd * safe_divide(Neg, Pos + Neg)
            Sb1 = safe_divide(bp, Z1)
            Sb2 = safe_divide(bn, Z2)
            Cb1 = px * self.gradprop(Z1, px, Sb1)[0]
            Cb2 = px * self.gradprop(Z2, px, Sb2)[0]
            return C1 + C4 + Cb1 + C2 + C3 + Cb2

        def backward(R_p, px, nx, pw, nw):
            Rp = f(R_p, pw, nw, px, nx)
            return Rp

        def redistribute(Rp_tmp):
            Rp = torch.clamp(Rp_tmp, min=0)
            Rn = torch.clamp(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        X = self.X
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  ## first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A = redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        return Rp


class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1, 2, 3], keepdim=True)) * torch.ne(R, 0).type(
                R.type())
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=[1, 2, 3], keepdim=True) - R.sum(dim=[1, 2, 3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            if w1.shape[2] == 1:
                xabs = self.X.abs()
                wabs = self.weight.abs()
                Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.stride, padding=self.padding) * R_nonzero
                S = safe_divide(R, Zabs)
                C = xabs * self.gradprop(Zabs, xabs, S)[0]

                return C
            else:
                R_nonzero = R.ne(0).type(R.type())
                Za1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding) * R_nonzero
                Za2 = - F.conv2d(x1, w2, bias=None, stride=self.stride, padding=self.padding) * R_nonzero

                Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.stride, padding=self.padding) * R_nonzero
                Zb2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding) * R_nonzero

                C1 = pos_prop(R, Za1, Za2, x1)
                C2 = pos_prop(R, Zb1, Zb2, x2)

                return C1 + C2

        def backward(R_p, px, nx, pw, nw):
            Rp = f(R_p, pw, nw, px, nx)
            return Rp

        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:
            Rp = backward(R_p, px, nx, pw, nw)
        return Rp


class MultiheadAttention(RelProp):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.softmax = Softmax(dim=-1)

        self.einsum1 = einsum('bid,bjd->bij')
        self.einsum2 = einsum('bij,bjd->bid')

        self._register_load_state_dict_pre_hook(MultiheadAttention._pre_load_state_dict)

        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn(self, attn):
        self.attn = attn

    def get_attn(self):
        return self.attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    @staticmethod
    def _pre_load_state_dict(state_dict: OrderedDict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        w = state_dict[prefix + 'in_proj_weight']
        b = state_dict[prefix + 'in_proj_bias']

        embed_dim = w.shape[1]

        state_dict[prefix + 'q_proj.weight'] = w[:embed_dim]
        state_dict[prefix + 'q_proj.bias'] = b[:embed_dim]

        state_dict[prefix + 'k_proj.weight'] = w[embed_dim:2*embed_dim]
        state_dict[prefix + 'k_proj.bias'] = b[embed_dim:2*embed_dim]

        state_dict[prefix + 'v_proj.weight'] = w[2*embed_dim:]
        state_dict[prefix + 'v_proj.bias'] = b[2*embed_dim:]

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len, _, _ = key.size()

        self.tgt_len = tgt_len
        self.src_len = src_len
        self.bsz = bsz

        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        self.head_dim = head_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)  # BHxSxD

        # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = self.einsum1([q, k])  # BHxTxS

        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.dropout(attn_output_weights)

        self.save_attn(attn_output_weights)
        attn_output_weights.register_hook(self.save_attn_gradients)

        # attn_output = torch.bmm(attn_output_weights, v)
        attn_output = self.einsum2([attn_output_weights, v])  # BHxTxD

        #  BHxTxD -> TxBHxD -> TxBxHD
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

    def relprop(self, cam_attn_output, alpha, **kwargs):
        cam_attn_output = self.out_proj.relprop(cam_attn_output, alpha, **kwargs)
        cam_attn_output = cam_attn_output.view(self.tgt_len, self.bsz*self.num_heads, self.head_dim).transpose(0, 1)
        cam_attn_output_weights, cam_v = self.einsum2.relprop(cam_attn_output, alpha, **kwargs)
        cam_attn_output_weights /= 2
        cam_v /= 2
        self.save_attn_cam(cam_attn_output_weights)
        cam_attn_output_weights = self.dropout.relprop(cam_attn_output_weights, alpha, **kwargs)
        cam_attn_output_weights = self.softmax.relprop(cam_attn_output_weights, alpha, **kwargs)
        cam_q, cam_k = self.einsum1.relprop(cam_attn_output_weights, alpha, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_v = cam_v.transpose(0, 1).view(self.src_len, self.bsz, self.num_heads*self.head_dim)
        cam_k = cam_k.transpose(0, 1).view(self.src_len, self.bsz, self.num_heads*self.head_dim)
        cam_q = cam_q.transpose(0, 1).view(self.tgt_len, self.bsz, self.num_heads*self.head_dim)

        pre_cam_v = cam_v.min() == cam_v.max() == 0
        cam_v = self.v_proj.relprop(cam_v, alpha, **kwargs)
        cam_k = self.k_proj.relprop(cam_k, alpha, **kwargs)
        cam_q = self.q_proj.relprop(cam_q, alpha, **kwargs)

        if cam_v.min() == cam_v.max() == 0 and not pre_cam_v:
            cam_k_sum = cam_k.sum()
            cam_q_sum = cam_q.sum()
            cam_k_fact = safe_divide(cam_k_sum.abs(), cam_k_sum.abs() + cam_q_sum.abs()) * cam_attn_output.sum()
            cam_q_fact = safe_divide(cam_q_sum.abs(), cam_k_sum.abs() + cam_q_sum.abs()) * cam_attn_output.sum()

            cam_k = cam_k * safe_divide(cam_k_fact, cam_k.sum())
            cam_q = cam_q * safe_divide(cam_q_fact, cam_q.sum())

        return cam_q, cam_k, cam_v
