import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

EPS = 1e-6

class IdentityQuantizer(nn.Module):
    def __init__(self, bitwidth=32, per_channel=False):
        super().__init__()
        self.bitwidth = bitwidth
        self.w_bits = bitwidth
        self.a_bits = bitwidth

    def forward(self, x):
        return x


class SymmetricQuantizerFixedAlphaFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        alpha = alpha.to(x.device)

        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.k = k

        y = x.clamp(min=-alpha, max=alpha)
        scale = (2 ** (k - 1) - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        mask = (x >= -ctx.alpha) & (x <= ctx.alpha)
        grad_x = grad_output * mask.float()
        return grad_x, None, None


class SymmetricQuantizerFixedAlpha(nn.Module):
    def __init__(self, bitwidth=8, init_method="max", percentile=99.9, per_channel=False):
        super().__init__()
        self.k = bitwidth
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.init_done = False
        self.init_method = init_method
        self.percentile = percentile
        self.per_channel = per_channel

    def init_alpha(self, x):
        x_detach = x.detach()

        if self.per_channel:
            if x_detach.ndim == 2:
                x_abs = x_detach.abs()
                if self.init_method == "max":
                    alpha = x_abs.max(dim=1, keepdim=True).values
                else:
                    alpha = torch.quantile(
                        x_abs, self.percentile / 100.0, dim=1, keepdim=True
                    )
            else:
                x_abs = x_detach.abs()
                if self.init_method == "max":
                    alpha = x_abs.amax(
                        dim=tuple(range(x_abs.ndim - 1)), keepdim=True
                    )
                else:
                    x_flat = x_abs.flatten(start_dim=0, end_dim=-2)
                    alpha = torch.quantile(
                        x_flat, self.percentile / 100.0, dim=0, keepdim=True
                    )
            alpha = alpha.clamp(min=1e-6)
        else:
            x_abs = x_detach.abs().flatten()
            if self.init_method == "max":
                alpha = x_abs.max()
            else:
                alpha = torch.quantile(x_abs, self.percentile / 100.0)
            alpha = max(alpha, 1e-6)

        return alpha

    def forward(self, x):
        if not self.init_done:
            with torch.no_grad():
                self.alpha.data.copy_(
                    self.init_alpha(x).to(self.alpha.device)
                )
            self.init_done = True

        alpha = self.alpha.clamp(min=1e-4)
        return SymmetricQuantizerFixedAlphaFunction.apply(x, alpha, self.k)

class ActFn(Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)

        alpha_safe = alpha.clamp(min=1e-6)
        y = torch.clamp(x, min=0.0, max=alpha_safe)
        scale = (2 ** k - 1) / alpha_safe
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha = ctx.saved_tensors

        grad_x = dLdy_q * ((x >= 0) & (x <= alpha)).float()
        grad_alpha = (dLdy_q * (x > alpha).float()).sum().unsqueeze(0)

        return grad_x, grad_alpha, None


class PACTActivationQuantizer(nn.Module):
    def __init__(self, bitwidth=8, init_batches=20):
        super().__init__()
        self.k = bitwidth
        self.init_batches = init_batches
        self.init_state = 0
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.init_state < self.init_batches:
            with torch.no_grad():
                a = torch.quantile(x.detach(), 0.999)
                self.alpha.data = a.clamp(min=1e-6)
                self.init_state += 1

        return ActFn.apply(x, self.alpha, self.k)

class Round(Function):
    @staticmethod
    def forward(ctx, input):
        sign = torch.sign(input)
        return sign * torch.floor(torch.abs(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp

        w_q = Round.apply(((weight - beta) / alpha).clamp(Qn, Qp))
        return w_q * alpha + beta

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other

        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger

        grad_alpha = (
            (smaller * Qn + bigger * Qp +
             between * Round.apply(q_w) - between * q_w)
            * grad_weight * g
        ).sum().unsqueeze(0)

        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(0)
        grad_weight = between * grad_weight

        return grad_weight, grad_alpha, None, None, None, grad_beta


class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, bitwidth, all_positive=False, batch_init=20):
        super().__init__()

        self.a_bits = bitwidth
        self.all_positive = all_positive
        self.batch_init = batch_init

        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** bitwidth - 1
        else:
            self.Qn = -2 ** (bitwidth - 1)
            self.Qp = 2 ** (bitwidth - 1) - 1

        self.disabled = self.Qp <= 0 
        self.s = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.init_state = 0

    def forward(self, activation):
        if self.disabled or self.a_bits == 32:
            return activation

        numel = max(activation.numel(), 1)
        Qp_safe = max(self.Qp, 1)

        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(numel * Qp_safe)

            mina = activation.detach().min()
            maxa = activation.detach().max()

            scale = (maxa - mina) / max(self.Qp - self.Qn, 1)
            self.s.data = scale.clamp(min=1e-6)
            self.beta.data = mina - self.s.data * self.Qn

            self.init_state += 1

        elif self.init_state < self.batch_init:
            mina = activation.detach().min()
            maxa = activation.detach().max()

            scale = (maxa - mina) / max(self.Qp - self.Qn, 1)
            scale = scale.clamp(min=1e-6)

            self.s.data = 0.9 * self.s.data + 0.1 * scale
            self.beta.data = 0.9 * self.beta.data + 0.1 * (
                mina - self.s.data * self.Qn
            )

            self.init_state += 1

        return ALSQPlus.apply(
            activation,
            self.s.clamp(min=1e-6),
            self.g,
            self.Qn,
            self.Qp,
            self.beta,
        )


class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, bitwidth=8, per_channel=False, init_batches=20):
        super().__init__()

        self.w_bits = bitwidth
        self.per_channel = per_channel
        self.init_batches = init_batches
        self.init_state = 0

        self.Qn = -2 ** (bitwidth - 1)
        self.Qp = 2 ** (bitwidth - 1) - 1

        self.disabled = self.Qp <= 0
        self.s = nn.Parameter(torch.ones(1))

    def forward(self, w):
        if self.disabled or self.w_bits == 32:
            return w

        Qp_safe = max(self.Qp, 1)
        numel = max(w.numel(), 1)

        if self.init_state < self.init_batches:
            with torch.no_grad():
                if self.per_channel:
                    w_ = w.detach().view(w.size(0), -1)
                    s_init = w_.abs().mean(dim=1) * 2 / math.sqrt(Qp_safe)
                else:
                    s_init = w.detach().abs().mean() * 2 / math.sqrt(Qp_safe)

                self.s.data = s_init.clamp(min=1e-6)
                self.init_state += 1

        g = 1.0 / math.sqrt(numel * Qp_safe)
        s = (self.s.clamp(min=1e-6) * g).detach() + self.s.clamp(min=1e-6) * (1 - g)

        return ((w / s).clamp(self.Qn, self.Qp).round()) * s
class SignedPACTFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, bits):
        Qp = 2 ** (bits - 1) - 1
        alpha_v = alpha.view(1, 1, -1)
        ctx.save_for_backward(x, alpha_v)
        y = torch.clamp(x, -alpha_v, alpha_v)
        scale = Qp / alpha_v
        return torch.round(y * scale) / scale

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha_v = ctx.saved_tensors
        inside = (x.abs() <= alpha_v).float()
        grad_x = grad_out * inside
        grad_alpha = (grad_out * torch.sign(x) * (x.abs() > alpha_v)).sum(dim=(0, 1))
        return grad_x, grad_alpha, None
class SignedPACT(nn.Module):
    def __init__(self, bits=8, init_alpha=16.0):
        super().__init__()
        self.bits = bits
        self.bits_orig = bits
        self.init_alpha = init_alpha
        self.alpha = None
    def forward(self, x):
        current_bits = getattr(self, 'bits', 8)
        
        if current_bits == 32:
            return x

        feat = x.shape[-1]
        if self.alpha is None or self.alpha.numel() != feat:
            self.alpha = nn.Parameter(torch.ones(feat, device=x.device) * self.init_alpha)

        self.alpha.data.clamp_(min=EPS)
        return SignedPACTFn.apply(x, self.alpha, current_bits)
