import paddle
import paddle.nn as nn
import paddle.nn.functional as F


__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2D', 'Linear', 'MaxPool2D',
           'AdaptiveAvgPool2D', 'AvgPool2D', 'Conv2D', 'Sequential', 'safe_divide']

ZERO_TENSOR = paddle.to_tensor(0.)


def safe_divide(a, b):
    return a / (b + b.equal(ZERO_TENSOR).astype(b.dtype) * 1e-9) * b.not_equal(ZERO_TENSOR).astype(b.dtype)


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.stop_gradient = False
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.stop_gradient = False

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Layer):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_post_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = paddle.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R

    def m_relprop(self, R, pred,  alpha):
        return R

    def RAP_relprop(self, R_p):
        return R_p


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)[0]

        if paddle.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C)
            outputs.append(self.X[1] * C)
        else:
            outputs = self.X * (C)
        return outputs

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)[0]
            if paddle.is_tensor(self.X) == False:
                Rp = []
                Rp.append(self.X[0] * Cp)
                Rp.append(self.X[1] * Cp)
            else:
                Rp = self.X * (Cp)
            return Rp
        if paddle.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class ReLU(nn.ReLU, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2D(nn.MaxPool2D, RelPropSimple):
    pass


class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2D, RelPropSimple):
    pass


class AvgPool2D(nn.AvgPool2D, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return paddle.add(*inputs)


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
                Spp.append(safe_divide(paddle.clip(rp, min=0), z))
                Spn.append(safe_divide(paddle.clip(rp, max=0), z))

            Cpp = self.gradprop(Z, self.X, Spp)[0]
            Cpn = self.gradprop(Z, self.X, Spn)[0]

            Rp = self.X * (Cpp * Cpn)

            return Rp
        if paddle.is_tensor(R_p) == False:
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
        return paddle.concat(inputs, dim)

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
        if paddle.is_tensor(R_p) == False:
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
        for m in reversed(self._sub_layers.values()):
            R = m.relprop(R, alpha)
        return R

    def RAP_relprop(self, Rp):
        for m in reversed(self._sub_layers.values()):
            Rp = m.RAP_relprop(Rp)
        return Rp


class BatchNorm2D(nn.BatchNorm2D, RelProp):
    def relprop(self, R, alpha):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self._variance.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self._epsilon).pow(0.5))
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
                (self._variance.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))

            if paddle.is_tensor(self.bias):
                bias = self.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(
                    bias * R_p.not_equal(ZERO_TENSOR).astype(self.bias.dtype),
                    R_p.not_equal(ZERO_TENSOR).astype(
                        self.bias.dtype).sum(dim=[2, 3], keepdim=True)
                )
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if paddle.is_tensor(self.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if paddle.is_tensor(R_p) == False:
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
        beta = alpha - 1
        pw = paddle.clip(self.weight, min=0)  # positive weight
        nw = paddle.clip(self.weight, max=0)  # negative weight
        px = paddle.clip(self.X, min=0)       # positive x
        nx = paddle.clip(self.X, max=0)       # negative x

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
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
            R_nonzero = paddle.not_equal(R, ZERO_TENSOR).astype(R.dtype)
            shift = safe_divide(R_val, paddle.sum(
                R_nonzero, dim=-1, keepdim=True)) * paddle.not_equal(R, ZERO_TENSOR).astype(R.dtype)
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = paddle.clip(R, min=0)
            R_neg = paddle.clip(R, max=0)
            S1 = safe_divide(
                (R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide(
                (R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n

            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=-1, keepdim=True) -
                          R.sum(dim=-1, keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.not_equal(ZERO_TENSOR).astype(R.dtype)
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
            # dealing bias
            # if paddle.is_tensor(self.bias):
            #     bias_p = self.bias * R_p.not_equal(ZERO_TENSOR).astype(self.bias.dtype)
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if paddle.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp

        def redistribute(Rp_tmp):
            Rp = paddle.clip(Rp_tmp, min=0)
            Rn = paddle.clip(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * \
                (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * \
                (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3

        pw = paddle.clip(self.weight, min=0)
        nw = paddle.clip(self.weight, max=0)
        X = self.X
        px = paddle.clip(X, min=0)
        nx = paddle.clip(X, max=0)
        if paddle.is_tensor(R_p) == True and R_p.max() == 1:  # first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A = redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        return Rp


class Conv2D(nn.Conv2D, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.shape[2] - \
            ((Z.shape[2] - 1) * self._stride[0] -
             2 * self._padding + self._kernel_size[0])

        return F.conv2d_transpose(
            DY, weight,
            stride=self._stride, padding=self._padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = paddle.clip(self.weight, min=0)
            nw = paddle.clip(self.weight, max=0)
            X = self.X
            # print(X.shape)  # [1, 3, 224, 224]
            L = self.X * 0 + \
                paddle.min(paddle.min(paddle.min(self.X, axis=1, keepdim=True),
                                      axis=2, keepdim=True),
                           axis=3, keepdim=True)
            H = self.X * 0 + \
                paddle.max(paddle.max(paddle.max(self.X, axis=1, keepdim=True),
                                      axis=2, keepdim=True),
                           axis=3, keepdim=True)
            Za = F.conv2d(X, self.weight, bias=None, stride=self._stride, padding=self._padding) - \
                F.conv2d(L, pw, bias=None, stride=self._stride, padding=self._padding) - \
                F.conv2d(H, nw, bias=None, stride=self._stride,
                         padding=self._padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * \
                self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = paddle.clip(self.weight, min=0)
            nw = paddle.clip(self.weight, max=0)
            px = paddle.clip(self.X, min=0)
            nx = paddle.clip(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None,
                              stride=self._stride, padding=self._padding)
                Z2 = F.conv2d(x2, w2, bias=None,
                              stride=self._stride, padding=self._padding)
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
            R_nonzero = paddle.not_equal(R, ZERO_TENSOR).astype(R.dtype)
            shift = safe_divide(R_val, paddle.sum(
                R_nonzero, dim=[1, 2, 3], keepdim=True)) * paddle.not_equal(R, ZERO_TENSOR).astype(R.dtype)
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = paddle.clip(R, min=0)
            R_neg = paddle.clip(R, max=0)
            S1 = safe_divide(
                (R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide(
                (R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(
                C, C.sum(dim=[1, 2, 3], keepdim=True) - R.sum(dim=[1, 2, 3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.not_equal(ZERO_TENSOR).astype(R.dtype)
            Za1 = F.conv2d(x1, w1, bias=None, stride=self._stride,
                           padding=self.padding) * R_nonzero
            Za2 = - F.conv2d(x1, w2, bias=None, stride=self._stride,
                             padding=self.padding) * R_nonzero

            Zb1 = - F.conv2d(x2, w1, bias=None, stride=self._stride,
                             padding=self.padding) * R_nonzero
            Zb2 = F.conv2d(x2, w2, bias=None, stride=self._stride,
                           padding=self.padding) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)
            return C1 + C2

        def backward(R_p, px, nx, pw, nw):

            # if paddle.is_tensor(self.bias):
            #     bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            #     bias_p = safe_divide(bias * R_p.ne(0).astype(self.bias.dtype),
            #                          R_p.ne(0).astype(self.bias.dtype).sum(dim=[2, 3], keepdim=True))
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if paddle.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp

        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                paddle.min(paddle.min(paddle.min(X,
                                                 dim=1, keepdim=True),
                                      dim=2, keepdim=True),
                           dim=3, keepdim=True)
            H = X * 0 + \
                paddle.max(paddle.max(paddle.max(X,
                                                 dim=1, keepdim=True),
                                      dim=2, keepdim=True),
                           dim=3, keepdim=True)
            Za = F.conv2d(X, self.weight, bias=None, stride=self._stride, padding=self._padding) - \
                F.conv2d(L, pw, bias=None, stride=self._stride, padding=self._padding) - \
                F.conv2d(H, nw, bias=None, stride=self._stride,
                         padding=self._padding)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * \
                self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp
        pw = paddle.clip(self.weight, min=0)
        nw = paddle.clip(self.weight, max=0)
        px = paddle.clip(self.X, min=0)
        nx = paddle.clip(self.X, max=0)

        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:
            Rp = backward(R_p, px, nx, pw, nw)
        return Rp
