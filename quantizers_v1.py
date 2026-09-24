import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityQuantizer(nn.Module):
    def __init__(self, bitwidth: int = 32, **_: object) -> None:
        super().__init__()
        self.bitwidth = bitwidth
        self.enable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class BlockUniformWeightQuantizer(nn.Module):
    def __init__(self, bitwidth: int = 8, block_size: int = 128, eps: float = 1e-6, **_: object) -> None:
        super().__init__()
        if bitwidth < 2:
            raise ValueError(f"bitwidth must be >= 2, got {bitwidth}")
        self.bitwidth = bitwidth
        self.block_size = block_size
        self.eps = eps
        self.enable = False
        self.qn = -(2 ** (bitwidth - 1))
        self.qp = 2 ** (bitwidth - 1) - 1
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return w
        out_f, in_f = w.shape
        pad = (self.block_size - in_f % self.block_size) % self.block_size
        if pad:
            w = F.pad(w, (0, pad))
        w_blk = w.view(out_f, -1, self.block_size)
        absmax = w_blk.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / max(self.qp, 1)) * torch.exp(self.log_scale)
        scale = scale.clamp_min(self.eps)

        q = w_blk / scale
        q_bar = q.clamp(self.qn, self.qp)
        q_hat = q_bar + (q_bar.round() - q_bar).detach()
        w_q = (q_hat * scale).view(out_f, -1)
        if pad:
            w_q = w_q[:, :-pad]
        return w_q

class NF4WeightQuantizer(nn.Module):
    def __init__(self, block_size: int = 64, eps: float = 1e-6, **_: object) -> None:
        super().__init__()
        self.block_size = block_size
        self.eps = eps
        self.enable = False
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "nf4_codebook",
            torch.tensor(
                [
                    -1.0,
                    -0.6961928,
                    -0.52507305,
                    -0.39491749,
                    -0.28444138,
                    -0.18477343,
                    -0.09105004,
                    0.0,
                    0.0795803,
                    0.1609302,
                    0.2461123,
                    0.33791524,
                    0.44070983,
                    0.5626170,
                    0.72295684,
                    1.0,
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return w

        out_f, in_f = w.shape
        pad = (self.block_size - in_f % self.block_size) % self.block_size
        if pad:
            w = F.pad(w, (0, pad))

        w_blk = w.view(out_f, -1, self.block_size)
        w_flat = w_blk.reshape(-1, self.block_size)

        absmax = w_flat.abs().amax(dim=1, keepdim=True)
        scale = absmax * torch.exp(self.log_scale)
        scale = scale.clamp_min(self.eps)

        w_norm = (w_flat / scale).clamp(-1.0, 1.0)
        dist = torch.abs(w_norm.unsqueeze(-1) - self.nf4_codebook.view(1, 1, -1))
        idx = dist.argmin(dim=-1)
        w_q_norm = self.nf4_codebook[idx]
        w_q_norm = w_norm + (w_q_norm - w_norm).detach()

        w_q = (w_q_norm * scale).view(out_f, -1)
        if pad:
            w_q = w_q[:, :-pad]
        return w_q


class AsymmetricPACT(nn.Module):
    def __init__(
        self,
        bitwidth: int = 8,
        init_alpha_low: float = -1.0,
        init_alpha_high: float = 1.0,
        eps: float = 1e-6,
        **_: object,
    ) -> None:
        super().__init__()
        if bitwidth < 2:
            raise ValueError(f"bitwidth must be >= 2, got {bitwidth}")
        self.bitwidth = bitwidth
        self.eps = eps
        self.enable = False
        self.alpha_low = nn.Parameter(torch.tensor(float(init_alpha_low)))
        self.alpha_delta = nn.Parameter(torch.tensor(float(max(init_alpha_high - init_alpha_low, 1e-3))))
        self.register_buffer("init_done", torch.tensor(False, dtype=torch.bool))

    def high(self) -> torch.Tensor:
        return self.alpha_low + self.alpha_delta.abs().clamp_min(1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x

        if self.training and not bool(self.init_done):
            with torch.no_grad():
                lo = x.detach().amin()
                hi = x.detach().amax()
                self.alpha_low.copy_(lo)
                self.alpha_delta.copy_((hi - lo).clamp_min(1e-3))
                self.init_done.fill_(True)
        lo = self.alpha_low
        hi = self.high()
        levels = 2 ** self.bitwidth - 1
        step = ((hi - lo) / max(levels, 1)).clamp_min(self.eps)
        x_bar = torch.clamp(x, lo, hi)
        q = torch.round((x_bar - lo) / step)
        x_hat = q * step + lo
        return x_bar + (x_hat - x_bar).detach()

class DynamicTokenQuantizer(nn.Module):
    def __init__(self, bitwidth: int = 8, use_softmax: bool = False, eps: float = 1e-6, **_: object) -> None:
        super().__init__()
        if bitwidth < 2:
            raise ValueError(f"bitwidth must be >= 2, got {bitwidth}")
        self.bitwidth = bitwidth
        self.use_softmax = use_softmax
        self.eps = eps
        self.enable = False
        self.qp = 2 ** (bitwidth - 1) - 1
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x

        abs_x = x.abs()
        if self.use_softmax:
            weights = torch.softmax(abs_x, dim=-1)
            token_absmax = (weights * abs_x).sum(dim=-1, keepdim=True)
        else:
            token_absmax = abs_x.amax(dim=-1, keepdim=True)

        scale = (token_absmax / max(self.qp, 1)) * torch.exp(self.log_scale)
        scale = scale.clamp_min(self.eps)

        q = x / scale
        q_bar = q.clamp(-self.qp, self.qp)
        q_hat = q_bar + (q_bar.round() - q_bar).detach()
        return q_hat * scale

def round_ste(x):
    return (torch.round(x) - x).detach() + x

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, num_bits, per_channel=True, channel_dim=0):
        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.Qn = -2 ** (num_bits - 1)
        self.Qp = 2 ** (num_bits - 1) - 1
        self.s = nn.Parameter(torch.ones(1))
        self.register_buffer('init_done', torch.tensor(False))

    def forward(self, x):
        if not self.init_done:
            with torch.no_grad():
                if self.per_channel:
                    dims = list(range(x.dim()))
                    dims.remove(self.channel_dim)
                    max_val = x.abs().amax(dim=dims, keepdim=True)
                    self.s.data = (max_val / self.Qp).detach()
                else:
                    self.s = nn.Parameter(x.abs().max() / self.Qp)
            self.init_done.fill_(True)
        if self.per_channel:
            N = x.numel() / x.shape[self.channel_dim]
        else:
            N = x.numel()
        g = 1.0 / math.sqrt(N * self.Qp)        
        s = grad_scale(self.s, g)
        s = torch.clamp(s, min=1e-8)
        x_scaled = x / s
        x_clamped = torch.clamp(x_scaled, self.Qn, self.Qp)
        x_rounded = round_ste(x_clamped)
        
        return x_rounded * s


class LSQPlusActQuantizer(nn.Module):
    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits
        self.Qn = 0
        self.Qp = 2 ** num_bits - 1
        self.s = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.register_buffer('init_done', torch.tensor(False))

    def forward(self, x):
        if not self.init_done:
            with torch.no_grad():
                min_val = x.min()
                max_val = x.max()
                scale = (max_val - min_val) / (self.Qp - self.Qn)
                scale = torch.clamp(scale, min=1e-8)               
                self.s.data.fill_(scale.item())
                self.beta.data.fill_(min_val.item())
            self.init_done.fill_(True)
        g = 1.0 / math.sqrt(x.numel() * self.Qp)        
        s = grad_scale(self.s, g)
        beta = grad_scale(self.beta, g)        
        s = torch.clamp(s, min=1e-8)
        x_scaled = (x - beta) / s
        x_clamped = torch.clamp(x_scaled, self.Qn, self.Qp)
        x_rounded = round_ste(x_clamped)
        
        return x_rounded * s + beta
