import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoForCausalLM
import json
import os
from typing import List, Dict
from torch.optim import AdamW, Adam
from quantizers_lsqplus_per_channel import LSQPlusActivationQuantizer, LSQPlusWeightQuantizer, PACTActivationQuantizer, SymmetricQuantizerFixedAlpha, IdentityQuantizer


class SmoothQuantRescale(nn.Module):
    def __init__(self, init_scale=1.0, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.eps = eps

    def weight_rescale(self, W):
        scale = torch.clamp(self.scale, min=self.eps)
        return W / scale

    def activation_rescale(self, X):
        scale = torch.clamp(self.scale, min=self.eps)
        return X * scale

def softmax_temperature(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def init_logits_bias(bits_list, prefer_bits=32):
    logits = torch.zeros(len(bits_list))
    if prefer_bits in bits_list:
        idx = bits_list.index(prefer_bits)
        logits[idx] = 0.0 
    return logits

class MixedPrecisionQATLinearEnhanced(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_heads: int,
                 hidden_dim:int,
                 quant_candidates_w: List[Dict],
                 quant_candidates_a: List[Dict],
                 init_temperature: float = 5.0,
                 min_temperature: float = 0.5,
                 anneal_rate: float = 0.99):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.d_head = self.hidden_dim // max(1, self.num_heads)
        self.quant_candidates_w = quant_candidates_w
        self.quant_candidates_a = quant_candidates_a
        self.quantizers_w = nn.ModuleList()

        for qc in quant_candidates_w:
            qtype = qc.get("qtype", "lsq")
            bitwidth = qc["bitwidth"]
            if qtype == "lsq":
                q = LSQPlusWeightQuantizer(bitwidth)
                setattr(q, "w_bits", bitwidth)
                self.quantizers_w.append(q)
            elif qtype == "uniform":
                q = SymmetricQuantizerFixedAlpha(bitwidth, per_channel=qc.get("per_channel", False))
                self.quantizers_w.append(q)
            elif qtype == "identity":
                q = IdentityQuantizer(bitwidth=bitwidth)
                self.quantizers_w.append(q)
            else:
                raise ValueError(f"Unknown weight quantizer type: {qtype}")
        self.bits_w_list = [qc["bitwidth"] for qc in quant_candidates_w]
        self.num_w = len(self.quantizers_w)
        self.logits_w = nn.Parameter(
            init_logits_bias(self.bits_w_list, prefer_bits=32)
        )
        self.quantizers_a = nn.ModuleList()
        for qc in quant_candidates_a:
            qtype = qc.get("qtype", "lsq")
            bitwidth = qc["bitwidth"]
            if qtype == "lsq":
                q = LSQPlusActivationQuantizer(bitwidth)
                setattr(q, "a_bits", bitwidth)
                self.quantizers_a.append(q)
            elif qtype == "pact":
                q = PACTActivationQuantizer(bitwidth)
                setattr(q, "a_bits", bitwidth)
                self.quantizers_a.append(q)
            elif qtype == "uniform":
                q = SymmetricQuantizerFixedAlpha(bitwidth)
                self.quantizers_a.append(q)
            elif qtype == "identity":
                q = IdentityQuantizer(bitwidth=bitwidth)
                self.quantizers_a.append(q)
            else:
                raise ValueError(f"Unknown activation quantizer type: {qtype}")
        self.bits_a_list = [qc["bitwidth"] for qc in quant_candidates_a]
        self.num_a = len(self.quantizers_a)
        self.logits_a = nn.Parameter(
            init_logits_bias(self.bits_a_list, prefer_bits=32)
        )
        self.linear = nn.Linear(in_features, out_features)
        self.rescale = SmoothQuantRescale()
        self.temperature = float(init_temperature)
        self.min_temperature = float(min_temperature)
        self.anneal_rate = float(anneal_rate)

    def forward(self, x: torch.Tensor):
            device = x.device
            if self.temperature is None:
                idx_w = self.bits_w_list.index(32)
                idx_a = self.bits_a_list.index(32)
                soft_alpha_w = F.one_hot(torch.tensor(idx_w, device=device), len(self.bits_w_list)).float()
                soft_alpha_a = F.one_hot(torch.tensor(idx_a, device=device), len(self.bits_a_list)).float()
            else:
                t = max(self.temperature, 1e-6)
                soft_alpha_w = F.softmax(self.logits_w / (t * 0.7), dim=-1)
                soft_alpha_a = F.softmax(self.logits_a / (t * 0.7), dim=-1)
            x_flat = x.reshape(-1, x.size(-1)) if x.dim() == 3 else x
            x_pre = self.rescale.activation_rescale(x_flat)
            W_pre = self.rescale.weight_rescale(self.linear.weight)
            q_x = 0.0
            current_act_error = 0.0
            x_target = x_pre.detach()
    
            for alpha_i, q in zip(soft_alpha_a, self.quantizers_a):
                x_q = q(x_pre)
                q_x = q_x + alpha_i * x_q
                
                if self.training and self.temperature is not None:
                    with torch.no_grad():
                        current_act_error = current_act_error + alpha_i * F.mse_loss(x_q.detach(), x_target)
            q_weight = 0.0
            current_weight_error = 0.0
            w_target = W_pre.detach()
    
            for alpha_i, q in zip(soft_alpha_w, self.quantizers_w):
                w_q = q(W_pre)
                q_weight = q_weight + alpha_i * w_q
                
                if self.training and self.temperature is not None:
                    with torch.no_grad():
                        current_weight_error = current_weight_error + alpha_i * F.mse_loss(w_q.detach(), w_target)
            if self.training:
                self.quant_error_act = current_act_error
                self.quant_error_weight = current_weight_error
            out_flat = F.linear(q_x, q_weight, self.linear.bias)
            if x.dim() == 3:
                return out_flat.reshape(x.shape[0], x.shape[1], self.out_features)
            return out_flat
    def layer_sensitivity_loss(self):
        if not self.training or self.temperature is None:
            return torch.zeros((), device=self.logits_w.device)
    
        temp = max(self.temperature, 1e-6)
    
        alpha_w = F.softmax(self.logits_w / temp, dim=-1)
        bits_w = torch.tensor(self.bits_w_list, device=alpha_w.device, dtype=alpha_w.dtype)
    
        min_bits = bits_w.min()
        bit_pressure = ((bits_w - min_bits) / 32.0)
    
        expected_pressure = (alpha_w * bit_pressure).sum()
    
        qe = self.quant_error_weight + self.quant_error_act
    
        return expected_pressure * qe

    def _force_select_min_bits_logits(self, logits: torch.Tensor, bits_list: List[int], device: torch.device):
        with torch.no_grad():
            bits = torch.tensor(bits_list, device=device)
            min_idx = int(torch.argmin(bits).item())
            forced = torch.full_like(logits, -1e9)
            forced[min_idx] = 1e9
            return forced

    def compute_layer_sensitivity(self, input_ids, labels, model, ce_loss_ref, subset_size: int = 32):
        model_was_training = model.training
        self_was_training = self.training

        device = next(self.parameters()).device
        if input_ids.size(0) > subset_size:
            idx = torch.randperm(input_ids.size(0), device=input_ids.device)[:subset_size]
            input_ids_sub = input_ids[idx].to(device)
            labels_sub = labels[idx].to(device)
        else:
            input_ids_sub = input_ids.to(device)
            labels_sub = labels.to(device)

        orig_logits_w = self.logits_w.data.clone()
        orig_logits_a = self.logits_a.data.clone()
        forced_w = self._force_select_min_bits_logits(self.logits_w.data, self.bits_w_list, device)
        forced_a = self._force_select_min_bits_logits(self.logits_a.data, self.bits_a_list, device)

        try:
            model.eval()
            self.eval()
            with torch.no_grad():
                self.logits_w.data.copy_(forced_w)
                self.logits_a.data.copy_(forced_a)

                outputs = model(input_ids=input_ids_sub, labels=labels_sub)
                loss_low_bits = outputs.loss.mean()

            sensitivity = (loss_low_bits - ce_loss_ref).item()
        finally:
            self.logits_w.data.copy_(orig_logits_w)
            self.logits_a.data.copy_(orig_logits_a)
            if model_was_training:
                model.train()
            if self_was_training:
                self.train()

        return sensitivity

    def linear_macs(self, avg_seq_len):
        return avg_seq_len * self.in_features * self.out_features

    def expected_linear_cost(self, avg_seq_len, num_heads=None):
        if self.temperature is None:
            return self.expected_linear_cost_base(avg_seq_len)
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
    
        p_w = F.softmax(self.logits_w.to(device), dim=-1)
        p_a = F.softmax(self.logits_a.to(device), dim=-1)
    
        base_macs = self.linear_macs(avg_seq_len)
    
        bits_w = torch.tensor(self.bits_w_list, device=device, dtype=dtype)
        bits_a = torch.tensor(self.bits_a_list, device=device, dtype=dtype)
    
        min_bw = bits_w.min().clamp(min=1.0)
        min_ba = bits_a.min().clamp(min=1.0)
        gamma_w = 1.2
        beta_a  = 0.7
        
        cost_matrix = (
            (bits_w.unsqueeze(1) / min_bw) ** gamma_w *
            (bits_a.unsqueeze(0) / min_ba) ** beta_a
        )


    
        prob_matrix = p_w.unsqueeze(1) * p_a.unsqueeze(0)
    
        expected_cost = (prob_matrix * cost_matrix).sum()
        return base_macs * expected_cost   


    def attention_macs(self, avg_seq_len):
        return ((self.num_heads * self.d_head) ** 2) * avg_seq_len

    def expected_attention_cost(self, avg_seq_len, num_heads=None):
        if self.temperature is None:
            return self.expected_attention_cost_base(avg_seq_len)
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
    
        p_w = F.softmax(self.logits_w.to(device), dim=-1)
        p_a = F.softmax(self.logits_a.to(device), dim=-1)
    
        base_macs = self.attention_macs(avg_seq_len)
    
        bits_w = torch.tensor(self.bits_w_list, device=device, dtype=dtype)
        bits_a = torch.tensor(self.bits_a_list, device=device, dtype=dtype)
    
        min_bw = bits_w.min().clamp(min=1.0)
        min_ba = bits_a.min().clamp(min=1.0)
        gamma_w = 1.2
        beta_a  = 0.7
        
        cost_matrix = (
            (bits_w.unsqueeze(1) / min_bw) ** gamma_w *
            (bits_a.unsqueeze(0) / min_ba) ** beta_a
        )


        prob_matrix = p_w.unsqueeze(1) * p_a.unsqueeze(0)
    
        expected_cost = (prob_matrix * cost_matrix).sum()
        return base_macs * expected_cost  


    def attention_macs_add(self, avg_seq_len):
        return 2 * self.num_heads * (avg_seq_len ** 2) * self.d_head

   
    def expected_linear_cost_base(self, avg_seq_len, num_heads=None):
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
    
        base_macs = self.linear_macs(avg_seq_len)
    
        gamma_w = 1.6 
        beta_a  = 1.0 
    
        cost_matrix = (32 ** gamma_w) * (32 ** beta_a)
    
        return base_macs * cost_matrix

    def expected_attention_cost_base(self, avg_seq_len, num_heads=None):
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
    
        base_macs = self.attention_macs(avg_seq_len)
    
        gamma_w = 1.6 
        beta_a  = 1.0 
    
        cost_matrix = (32 ** gamma_w) * (32 ** beta_a)
    
        return base_macs * cost_matrix

def compute_avg_seq_len(dataloader, pad_token_id, max_batches=200):
    total_tokens = 0
    total_seqs = 0
    for i, batch in enumerate(dataloader):
        ids = batch["input_ids"]
        mask = ids != pad_token_id
        total_tokens += mask.sum().item()
        total_seqs += ids.size(0)
        if i >= max_batches:
            break
    return total_tokens / max(total_seqs, 1)

def replace_linears_with_qat(
    module: nn.Module,
    quant_candidates_w: List[Dict],
    quant_candidates_a: List[Dict],
    init_temperature: float = 5.0,
    min_temperature: float = 0.5,
    anneal_rate: float = 0.99,
    num_heads: int = 12,
    hidden_dim: int = 768
):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in ["lm_head"] and "wte" not in name:
            in_f = child.in_features
            out_f = child.out_features

            mp = MixedPrecisionQATLinearEnhanced(
                in_features=in_f,
                out_features=out_f,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                quant_candidates_w=quant_candidates_w,
                quant_candidates_a=quant_candidates_a,
                init_temperature=init_temperature,
                min_temperature=min_temperature,
                anneal_rate=anneal_rate
            )

            mp.to(child.weight.device)
            mp.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                mp.linear.bias.data.copy_(child.bias.data)

            if child.training:
                mp.train()
            else:
                mp.eval()

            setattr(module, name, mp)
        else:
            replace_linears_with_qat(
                child,
                quant_candidates_w,
                quant_candidates_a,
                init_temperature,
                min_temperature,
                anneal_rate,
                num_heads,
                hidden_dim
            )

def precompute_layer_sensitivities(model: nn.Module, sample_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in sample_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.mean().item()
            num_batches += 1
            if num_batches >= 5: 
                break
    baseline_ce_loss = total_loss / max(num_batches, 1)
    sensitivities = {}
    with torch.no_grad():
        for batch in sample_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionQATLinearEnhanced) and name not in sensitivities:
                    sens = module.compute_layer_sensitivity(
                        input_ids=input_ids,
                        labels=labels,
                        model=model,
                        ce_loss_ref=baseline_ce_loss
                    )
                    sensitivities[name] = sens
            break 

    return sensitivities, baseline_ce_loss

def make_collate_fn(tokenizer):
    def collate_fn(batch):
        batch = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    return collate_fn




class TrainableLambda(nn.Module):
    def __init__(self, init_value=150):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_value, dtype=torch.float32)))

    @property
    def value(self):
        return torch.exp(self.log_lambda)

class BitBalanceController:
    def __init__(self, lambda_init=50, alpha=0.2,
                 temp_increase_factor=1.02,
                 temp_decrease_factor=0.995,
                 sensitivity_eps=1e-6,
                 sensitivity_clip=10.0):
        self.lambda_cost = float(lambda_init)
        self.alpha = float(alpha)
        self.prev_task_loss = None
        self.temp_increase_factor = float(temp_increase_factor)
        self.temp_decrease_factor = float(temp_decrease_factor)
        self.sensitivity_eps = sensitivity_eps
        self.sensitivity_clip = sensitivity_clip

    def update_lambda(self, task_loss: float):
        if self.prev_task_loss is None:
            self.prev_task_loss = float(task_loss)
            return self.lambda_cost
        diff = float(task_loss) - float(self.prev_task_loss)
        if diff > 0:
            self.lambda_cost *= 0.8
        else:
            self.lambda_cost *= 1.03
        self.lambda_cost = float(min(max(self.lambda_cost, 5.0), 150.0))
        self.prev_task_loss = float(task_loss)
        return self.lambda_cost

    def penalty_weight_bits(self, modules, module_names, device):
        penalties = []
    
        for m in modules:
            if m.temperature is None:
                continue
    
            temp = max(float(m.temperature), 1e-6)
            alpha_w = F.softmax(m.logits_w / temp, dim=-1)
            alpha_a = F.softmax(m.logits_a / temp, dim=-1)
    
            bits_w = torch.tensor(m.bits_w_list, device=device, dtype=alpha_w.dtype)
            bits_a = torch.tensor(m.bits_a_list, device=device, dtype=alpha_a.dtype)
    
            min_w = bits_w.min()
            min_a = bits_a.min()
            bits_penalty = (
                1.5 * (alpha_w * (bits_w - min_w)).sum() +
                0.5 * (alpha_a * (bits_a - min_a)).sum()
            ) / 32.0
    
            fp32_penalty = alpha_w[bits_w == 32].sum()
    
            penalties.append(bits_penalty + 2.0 * fp32_penalty)
    
        if len(penalties) == 0:
            return torch.zeros((), device=device, requires_grad=True)
    
        return self.alpha * torch.stack(penalties).mean()


    def adjust_temperature(
        self,
        modules: List[MixedPrecisionQATLinearEnhanced],
        task_loss: float
    ):
        if len(modules) == 0 or modules[0].temperature is None:
            self.prev_task_loss = float(task_loss)
            return
    
        if self.prev_task_loss is None:
            self.prev_task_loss = float(task_loss)
            return
    
        diff = float(task_loss) - float(self.prev_task_loss)
    
        for m in modules:
            if m.temperature is None:
                continue
    
            temp = max(float(m.temperature), 1e-6)
            pw = F.softmax(m.logits_w / temp, dim=-1)
            entropy = -(
                F.softmax(m.logits_w / temp, dim=-1) *
                F.log_softmax(m.logits_w / temp, dim=-1)
            ).sum()
    
            if entropy < 0.5:
                m.temperature = min(100.0, temp * 1.1)
    
            else:
                if diff > 0:
                    m.temperature = min(100.0, temp * self.temp_increase_factor)
                else:
                    m.temperature = max(
                        m.min_temperature,
                        temp * self.temp_decrease_factor
                    )
    
        self.prev_task_loss = float(task_loss)


def set_requires_grad(model: nn.Module, flag: bool, only_logits: bool = False):
    for name, p in model.named_parameters():
        if only_logits:
            if "logits" in name:
                p.requires_grad = flag
        else:
            p.requires_grad = flag

def extract_selected_quantizers(model: nn.Module):

    selected = {}

    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionQATLinearEnhanced):
            sw = F.softmax(module.logits_w.detach(), dim=-1)
            sa = F.softmax(module.logits_a.detach(), dim=-1)

            w_idx = int(torch.argmax(sw))
            a_idx = int(torch.argmax(sa))

            selected[name] = {
                "weight": {
                    "bits": module.bits_w_list[w_idx],
                    "qtype": module.quant_candidates_w[w_idx]["qtype"],
                    "prob": float(sw[w_idx])
                },
                "activation": {
                    "bits": module.bits_a_list[a_idx],
                    "qtype": module.quant_candidates_a[a_idx]["qtype"],
                    "prob": float(sa[a_idx])
                },
                "temperature": float(module.temperature)
            }

    return selected
def extract_selected_quantizers(model):
    selected = {}

    for name, module in model.named_modules():
        if not isinstance(module, MixedPrecisionQATLinearEnhanced):
            continue

        if module.temperature is None:
            w_bits = 32
            a_bits = 32
            w_idx = module.bits_w_list.index(32)
            a_idx = module.bits_a_list.index(32)
        else:
            sw = F.softmax(module.logits_w, dim=-1)
            sa = F.softmax(module.logits_a, dim=-1)
            w_idx = int(torch.argmax(sw))
            a_idx = int(torch.argmax(sa))
            w_bits = module.bits_w_list[w_idx]
            a_bits = module.bits_a_list[a_idx]

        selected[name] = {
            "weight": {
                "bits": w_bits,
                "qtype": module.quant_candidates_w[w_idx]["qtype"]
            },
            "activation": {
                "bits": a_bits,
                "qtype": module.quant_candidates_a[a_idx]["qtype"]
            },
            "temperature": module.temperature
        }

    return selected
def entropy_coeff(global_step, start, end, decay_steps):
    if global_step >= decay_steps:
        return end
    t = global_step / decay_steps
    return start * (1 - t) + end * t

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "EleutherAI/gpt-neo-125M"
    cost_ema = torch.tensor(1.0, device=device)
    penalty_ema = torch.tensor(0.1, device=device)
    ema_beta = 0.98

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLM.from_pretrained(model_name)

    quant_candidates_w = [
        {"bitwidth": 32, "qtype": "identity"},
        {"bitwidth": 16, "qtype": "lsq"},
        {"bitwidth": 8, "qtype": "lsq"},
        {"bitwidth": 4, "qtype": "lsq"},
        {"bitwidth": 2, "qtype": "lsq"},
        {"bitwidth": 16, "qtype": "uniform"},
        {"bitwidth": 8, "qtype": "uniform"},
        {"bitwidth": 4, "qtype": "uniform"},
        {"bitwidth": 2, "qtype": "uniform"}
    ]

    quant_candidates_a = [
        {"bitwidth": 32, "qtype": "identity"},
        {"bitwidth": 16, "qtype": "lsq"},
        {"bitwidth": 8, "qtype": "lsq"},
        {"bitwidth": 6, "qtype": "lsq"},
        {"bitwidth": 16, "qtype": "pact"},
        {"bitwidth": 8, "qtype": "pact"},
        {"bitwidth": 6, "qtype": "pact"},
        {"bitwidth": 16, "qtype": "uniform"},
        {"bitwidth": 8, "qtype": "uniform"},
        {"bitwidth": 6, "qtype": "uniform"},
    ]

    init_temperature = 30.0
    min_temperature = 10.0
    anneal_rate = 0.5
    entropy_start = 0.15     
    entropy_end   = 0.01
    entropy_decay_steps = 2000


    replace_linears_with_qat(
        model,
        quant_candidates_w,
        quant_candidates_a,
        init_temperature=init_temperature,
        min_temperature=min_temperature,
        anneal_rate=anneal_rate
    )
    
    if torch.cuda.device_count() >= 2:
        print("Using DataParallel on 2 GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1])
    
    model.to(device)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    dataset = dataset.shuffle(seed=42)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256
        )
    
    n_total = len(dataset)
    n_train = int(0.6 * n_total)
    
    train_ds = dataset.select(range(0, n_train))
    val_ds   = dataset.select(range(n_train, n_total))

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok   = val_ds.map(tokenize_fn, batched=True)
    
    train_tok.set_format(type="torch", columns=["input_ids"])
    val_tok.set_format(type="torch", columns=["input_ids"])
    
    collate = make_collate_fn(tokenizer)
    
    train_loader = DataLoader(
        train_tok,
        batch_size=8,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_tok,
        batch_size=8,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_iter = iter(val_loader)

    layer_sensitivities, baseline_ce_loss = precompute_layer_sensitivities(model, train_loader, device)

    print("Layer sensitivities precomputed:")
    for n, s in layer_sensitivities.items():
        print(f"{n}: {s:.6f}")

    trainable_lambda = nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32, device=device)))
    quant_params = []
    weight_params = []
    
    for n, p in model.named_parameters():
        if any(k in n for k in ["s", "alpha", "beta"]):
            quant_params.append(p)
        elif "logits" not in n:
            weight_params.append(p)
    
    optimizer_model = AdamW(
        [
            {"params": weight_params, "lr": 2e-5},
            {"params": quant_params,  "lr": 1e-4}
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    optimizer_bits = torch.optim.Adam(
        [p for n, p in model.named_parameters() if "logits" in n] + [trainable_lambda],
        lr=3e-4,
        betas=(0.9, 0.99)
    )
    lambda_cost = 19 
    lambda_ce_bits = 0.7
    num_epochs_qat = 3
    log_every = 50

    controller = BitBalanceController(lambda_init=lambda_cost, alpha=0.2)

    os.makedirs("qat_snapshots", exist_ok=True)
    os.makedirs("best_bits_steps", exist_ok=True)
    pad_id = tokenizer.pad_token_id
    avg_seq_len = compute_avg_seq_len(train_loader, pad_id)
    max_linear_cost = torch.tensor(0.0, device=device)
    max_attention_macs_add = torch.tensor(0.0, device=device)
    warmup_fp_steps  = 250     
    warmup_qat_steps = 500
    for name, module in model.named_modules():
        if not isinstance(module, MixedPrecisionQATLinearEnhanced):
            continue
            
        if "c_fc" in name or "c_proj" in name:
            max_linear_cost = max_linear_cost + module.expected_linear_cost_base(
            avg_seq_len=avg_seq_len)
        elif "attn" in name and any(k in name for k in ["q", "k", "v", "out"]):
            if "q" in name or "k" in name or "v" or "out" in name:
                max_linear_cost = max_linear_cost + module.expected_attention_cost_base(
                avg_seq_len=avg_seq_len)
            if "q" in name:
                max_attention_macs_add += module.attention_macs_add(
                    avg_seq_len=avg_seq_len)
    base_size = max_linear_cost + max_attention_macs_add
    global_step = 0
    for epoch in range(num_epochs_qat):
        model.train()
    
        for step, batch in enumerate(train_loader):
            global_step += 1
            if global_step < warmup_fp_steps:
                for m in model.modules():
                    if isinstance(m, MixedPrecisionQATLinearEnhanced):
                        m.temperature = None
                        m.logits_w.requires_grad_(False)
                        m.logits_a.requires_grad_(False)
    
            elif global_step < warmup_qat_steps:
                for m in model.modules():
                    if isinstance(m, MixedPrecisionQATLinearEnhanced):
                        m.temperature = init_temperature
                        m.logits_w.requires_grad_(True)
                        m.logits_a.requires_grad_(True)

    
            else:
                for m in model.modules():
                    if isinstance(m, MixedPrecisionQATLinearEnhanced):
                        m.logits_w.requires_grad_(True)
                        m.logits_a.requires_grad_(True)
    
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            input_ids = input_ids.long()
            labels = labels.long()

            for m in model.modules():
                if not isinstance(m, MixedPrecisionQATLinearEnhanced):
                    continue
    
            set_requires_grad(model, False)
            set_requires_grad(model, True, only_logits=False)
            set_requires_grad(model, False, only_logits=True)
    
            optimizer_model.zero_grad()
    
            outputs = model(input_ids=input_ids, labels=labels)
            train_ce_loss = outputs.loss.mean()

            val_ce_loss = train_ce_loss.detach()
            train_ce_loss.backward()
            optimizer_model.step()

            set_requires_grad(model, False)
            set_requires_grad(model, True, only_logits=True)

            optimizer_bits.zero_grad()

            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            
            val_input_ids = val_batch["input_ids"].to(device).long()
            val_labels    = val_batch["labels"].to(device).long()
            
            outputs = model(input_ids=val_input_ids, labels=val_labels)
            val_ce_loss = outputs.loss.mean()


            linear_cost = torch.tensor(0.0, device=device)
            attention_macs_add = torch.tensor(0.0, device=device)
                        
            avg_seq_len = input_ids.size(1)

            qat_modules = []
            qat_module_names = []
            
            
            for name, module in model.named_modules():
                if not isinstance(module, MixedPrecisionQATLinearEnhanced):
                    continue
            
                if "c_fc" in name or "c_proj" in name:
                    linear_cost = linear_cost + module.expected_linear_cost(
                        avg_seq_len=avg_seq_len)
                
                elif "attn" in name and any(k in name for k in ["q", "k", "v", "out"]):
                    if "q" in name or "k" in name or "v" or "out" in name:
                        linear_cost = linear_cost + module.expected_attention_cost(
                            avg_seq_len=avg_seq_len)
                    if "q" in name:
                        attention_macs_add += module.attention_macs_add(
                            avg_seq_len=avg_seq_len)

                    qat_modules.append(module)
                    qat_module_names.append(name)

            normalized_cost_reg = (linear_cost + attention_macs_add) / base_size
            curr_lambda = torch.tensor(controller.update_lambda(float(val_ce_loss.detach())),device=device)

            controller.adjust_temperature(qat_modules, float(val_ce_loss.detach()))
            penalty = controller.penalty_weight_bits(
                modules=qat_modules,
                module_names=qat_module_names,
                device=device
            )   
            lambda_cost = curr_lambda                 
            lambda_penalty = 0.3 * curr_lambda    

            with torch.no_grad():
                cost_ema = ema_beta * cost_ema + (1 - ema_beta) * normalized_cost_reg.detach()
                penalty_ema = ema_beta * penalty_ema + (1 - ema_beta) * penalty.detach()

            entropy_reg = 0.0
            for m in qat_modules:
                if m.temperature is None:
                    continue
            
                temp = max(m.temperature, 1e-6)
            
                pw = F.softmax(m.logits_w / temp, dim=-1)
                pa = F.softmax(m.logits_a / temp, dim=-1)
            
                entropy_w = - (pw * torch.log(pw + 1e-8)).sum()
                entropy_a = - (pa * torch.log(pa + 1e-8)).sum()
                entropy = 0.5 * entropy_w + 0.5 * entropy_a
                entropy_reg = entropy_reg + torch.clamp(1.2 - entropy, min=0.0)
            
            upper_loss = (
                val_ce_loss
                + lambda_cost * (normalized_cost_reg / (cost_ema + 1e-6))
                + lambda_penalty * (penalty / (penalty_ema + 1e-6))
            )



    
            upper_loss.backward()
            optimizer_bits.step()

            if step % log_every == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step}] "
                    f"train CE: {train_ce_loss.item():.6f} | "
                    f"val CE: {val_ce_loss.item():.6f} | "
                    f"cost: {float(normalized_cost_reg):.6f} | "
                    f"λ: {curr_lambda:.2e} | penalty: {penalty:.4f}"
                )
                for name, module in model.named_modules():
                    if isinstance(module, MixedPrecisionQATLinearEnhanced):
                        if module.temperature is None:
                            print(
                                f"[{name}] FP32 mode | T=None"
                            )
                        else:
                            temp = max(float(module.temperature), 1e-6)
                            pw = F.softmax(module.logits_w / temp, dim=-1)
                            bits = torch.tensor(module.bits_w_list, device=pw.device)
                            print(
                                f"[{name}] E[bits]={float((pw * bits).sum()):.2f}, "
                                f"T={temp:.2f}"
                            )

                        break
            
                selected = extract_selected_quantizers(model)
            
                print("Selected bits (first layers):")
                for i, (k, v) in enumerate(selected.items()):
                    print(
                        f"  {k}: "
                        f"W={v['weight']['bits']} ({v['weight']['qtype']}) | "
                        f"A={v['activation']['bits']} ({v['activation']['qtype']})"
                    )
            
                with open(f"best_bits_steps/selected_epoch{epoch}_step{step}.json", "w") as f:
                    json.dump(selected, f, indent=2)

if __name__ == "__main__":
    main()