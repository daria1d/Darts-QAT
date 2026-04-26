import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPTNeoForCausalLM
from quantizers import (
    IdentityQuantizer,
    BlockUniformWeightQuantizer,
    NF4WeightQuantizer,
    AsymmetricPACT,
    DynamicTokenQuantizer,
    LSQPlusWeightQuantizer,
    LSQPlusActQuantizer,
)

def init_logits_bias(bits_list: Sequence[int], prefer_bits: int = 32, init_bias: float = 4.0) -> torch.Tensor:
    logits = torch.full((len(bits_list),), 1.0, dtype=torch.float32)
    return logits

@dataclass(frozen=True)
class CandidateSpec:
    bitwidth: int
    qtype: str
    block_size: Optional[int] = None

class MixedPrecisionQATLinearEnhanced(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        hidden_dim: int,
        layer_name: str,
        quant_candidates_w: Sequence[Dict],
        quant_candidates_a: Sequence[Dict],
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.layer_name = layer_name
        self.num_heads = 12
        self.d_head = self.hidden_dim // max(1, self.num_heads)
        self.search_w = True
        self.search_a = True
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.quant_candidates_w = [dict(c) for c in quant_candidates_w]
        self.quant_candidates_a = [dict(c) for c in quant_candidates_a]
        self.quantizers_w = nn.ModuleList([self._build_quantizer(c, kind="weight") for c in quant_candidates_w])
        self.quantizers_a = nn.ModuleList([self._build_quantizer(c, kind="activation") for c in quant_candidates_a])
        self.bits_w_list = [int(c["bitwidth"]) for c in quant_candidates_w]
        self.bits_a_list = [int(c["bitwidth"]) for c in quant_candidates_a]
        self.logits_w = nn.Parameter(init_logits_bias(self.bits_w_list, prefer_bits=32))
        self.logits_a = nn.Parameter(init_logits_bias(self.bits_a_list, prefer_bits=32))
        self.register_buffer("last_error_penalty", torch.tensor(0.0, dtype=torch.float32))
        self.collect_error_stats = True
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _build_quantizer(cfg: Dict, kind: str) -> nn.Module:
        qtype = cfg["qtype"]
        bitwidth = int(cfg["bitwidth"])
        if qtype == "identity":
            return IdentityQuantizer(bitwidth=bitwidth)
        if kind == "weight":
            if qtype == "nf4":
                return NF4WeightQuantizer(block_size=int(cfg.get("block_size", 64)))
            if qtype == "block_uniform":
                return BlockUniformWeightQuantizer(bitwidth=bitwidth, block_size=int(cfg.get("block_size", 128)))
            if qtype == "lsq":
                return LSQPlusWeightQuantizer(num_bits=bitwidth, per_channel=True)
            raise ValueError(f"Unsupported weight quantizer: {qtype}")
        if kind == "activation":
            if qtype == "asym_pact":
                return AsymmetricPACT(bitwidth=bitwidth)
            if qtype == "dynamic_token":
                return DynamicTokenQuantizer(bitwidth=bitwidth)
            if qtype == "lsq":
                return LSQPlusActQuantizer(num_bits = bitwidth)
            raise ValueError(f"Unsupported activation quantizer: {qtype}")
        raise ValueError(f"Unknown quantizer kind: {kind}")

    def architecture_parameters(self) -> List[nn.Parameter]:
        return [self.logits_w, self.logits_a]

    def quantizer_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for q in list(self.quantizers_w) + list(self.quantizers_a):
            params.extend(list(q.parameters()))
        return params

    def soft_selection(self, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.search_w:
            p_w = F.softmax(self.logits_w / temperature, dim=-1)
        else:
            p_w = torch.zeros_like(self.logits_w)
            p_w[self.bits_w_list.index(32)] = 1.0
        if self.search_a:
            p_a = F.softmax(self.logits_a / temperature, dim=-1)
        else:
            p_a = torch.zeros_like(self.logits_a)
            p_a[self.bits_a_list.index(32)] = 1.0       
        return p_w, p_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_is_3d = x.dim() == 3
        x_flat = x.reshape(-1, x.size(-1)) if x_is_3d else x
        p_w, p_a = self.soft_selection()
        q_weight = 0
        for alpha, q in zip(p_w, self.quantizers_w):
            q_weight = q_weight + alpha * q(self.weight)    
        q_x = 0
        for alpha, q in zip(p_a, self.quantizers_a):
            q_x = q_x + alpha * q(x_flat)
        if self.collect_error_stats:
            ref_w = self.weight.detach()
            ref_x = x_flat.detach()
            err_w = F.mse_loss(q_weight, ref_w) / ref_w.pow(2).mean().clamp_min(1e-6)
            err_x = F.mse_loss(q_x, ref_x) / ref_x.pow(2).mean().clamp_min(1e-6)
            self.last_error_penalty = err_w + err_x
        else:
            self.last_error_penalty = q_weight.new_zeros(())
        out = F.linear(q_x, q_weight, self.bias)
        if x_is_3d:
            return out.view(x.shape[0], x.shape[1], self.out_features)
        return out

    def expected_weight_bits(self) -> torch.Tensor:
        if not any(q.enable for q in self.quantizers_w):
            return torch.tensor(32.0, device=self.weight.device, dtype=self.logits_w.dtype)
    
        probs = F.softmax(self.logits_w, dim=-1)
        bits = torch.tensor(self.bits_w_list, device=probs.device, dtype=probs.dtype)
        return (probs * bits).sum()
    
    def expected_activation_bits(self) -> torch.Tensor:
        if not any(q.enable for q in self.quantizers_a):
            return torch.tensor(32.0, device=self.weight.device, dtype=self.logits_a.dtype)
        probs = F.softmax(self.logits_a, dim=-1)
        bits = torch.tensor(self.bits_a_list, device=probs.device, dtype=probs.dtype)
        return (probs * bits).sum()

    def projection_bitops(self, avg_seq_len: torch.Tensor) -> torch.Tensor:
        macs = avg_seq_len * self.in_features * self.out_features
        return macs * self.expected_weight_bits() * self.expected_activation_bits()

    def weight_memory_bits(self) -> torch.Tensor:
        return torch.tensor(self.weight.numel(), device=self.weight.device, dtype=self.weight.dtype) * self.expected_weight_bits()

    def full_precision_projection_bitops(self, avg_seq_len: torch.Tensor) -> torch.Tensor:
        macs = avg_seq_len * self.in_features * self.out_features
        return macs * 32.0 * 32.0
        
    def compute_local_error(self, sample_x: torch.Tensor) -> torch.Tensor:
        p_w, p_a = self.soft_selection()
        q_w = sum(alpha * q(self.weight) for alpha, q in zip(p_w, self.quantizers_w))
        q_x = sum(alpha * q(sample_x) for alpha, q in zip(p_a, self.quantizers_a))
        err_w = torch.dist(q_w, self.weight.detach()) / self.weight.detach().norm().clamp_min(1e-6)
        err_x = torch.dist(q_x, sample_x.detach()) / sample_x.detach().norm().clamp_min(1e-6)
        return err_w + err_x
        
    def linear_macs(self, avg_seq_len):
        return avg_seq_len * self.in_features * self.out_features

    def expected_linear_cost(self, avg_seq_len, num_heads=None):
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
        p_w = F.softmax(self.logits_w.to(device), dim=-1)
        p_a = F.softmax(self.logits_a.to(device), dim=-1)
        base_macs = self.linear_macs(avg_seq_len)
        bits_w = torch.tensor(self.bits_w_list, device=device, dtype=dtype)
        bits_a = torch.tensor(self.bits_a_list, device=device, dtype=dtype)
        min_bw = bits_w.min().clamp(min=1.0)
        min_ba = bits_a.min().clamp(min=1.0)
        gamma_w = 2.2
        beta_a  = 1.1
        cost_matrix = (
            (bits_w.unsqueeze(1) / 32) ** gamma_w *
            (bits_a.unsqueeze(0) / 32) ** beta_a
        )
        prob_matrix = p_w.unsqueeze(1) * p_a.unsqueeze(0)
        expected_cost = (prob_matrix * cost_matrix).sum()
        return base_macs * expected_cost   

    def attention_macs(self, avg_seq_len):
        return ((self.num_heads * self.d_head) ** 2) * avg_seq_len

    def expected_attention_cost(self, avg_seq_len, num_heads=None):
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
        p_w = F.softmax(self.logits_w.to(device), dim=-1)
        p_a = F.softmax(self.logits_a.to(device), dim=-1)
        base_macs = self.attention_macs(avg_seq_len)
        bits_w = torch.tensor(self.bits_w_list, device=device, dtype=dtype)
        bits_a = torch.tensor(self.bits_a_list, device=device, dtype=dtype)
        gamma_w = 2.2
        beta_a  = 1.1 
        cost_matrix = (
            (bits_w.unsqueeze(1) / 32) ** gamma_w *
            (bits_a.unsqueeze(0) / 32) ** beta_a
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
        gamma_w = 2.2
        beta_a  = 1.1
        cost_matrix = (32 ** gamma_w) * (32 ** beta_a)
        return base_macs

    def expected_attention_cost_base(self, avg_seq_len, num_heads=None):
        device = next(self.parameters()).device
        dtype = self.logits_w.dtype
        base_macs = self.attention_macs(avg_seq_len)
        gamma_w = 2.2 
        beta_a  = 1.1 
        cost_matrix = (32 ** gamma_w) * (32 ** beta_a)
        return base_macs
        
@torch.no_grad()
def initialize_lsq_parameters(model: nn.Module, init_batch: Dict[str, torch.Tensor], device: torch.device):
    model.eval()
    inputs = init_batch["input_ids"].to(device)
    attention_mask = init_batch["attention_mask"].to(device)
    for m in model.modules():
        if isinstance(m, MixedPrecisionQATLinearEnhanced):
            m.search_w = False
            m.search_a = False
            for q in m.quantizers_w: q.enable = True
            for q in m.quantizers_a: q.enable = True
            for q in list(m.quantizers_w) + list(m.quantizers_a):
                if hasattr(q, 'init_done'):
                    q.init_done.fill_(False)
    model(input_ids=inputs, attention_mask=attention_mask) 
    model.train()
    
def compute_avg_seq_lenn(dataloader, pad_token_id, max_batches=200):
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

def compute_sampled_error_penalty(model: nn.Module, batch: Dict[str, torch.Tensor], num_samples: int = 128) -> torch.Tensor:
    penalties = []

    for m in model.modules():
        if isinstance(m, MixedPrecisionQATLinearEnhanced):
            device = next(m.parameters()).device
            fake_input = torch.randn(num_samples, m.in_features, device=device)
            penalties.append(m.compute_local_error(fake_input))

    return torch.stack(penalties).mean()

def set_error_collection(model: nn.Module, flag: bool) -> None:
    for m in model.modules():
        if isinstance(m, MixedPrecisionQATLinearEnhanced):
            m.collect_error_stats = flag
            
def layer_role_from_name(full_name: str) -> str:
    if full_name.endswith(("q_proj", "k_proj", "v_proj", "out_proj")):
        return "attn_proj"
    if full_name.endswith(("c_fc", "c_proj")):
        return "mlp"
    return "other"

def replace_linears_with_qat(
    module: nn.Module,
    quant_candidates_w: Sequence[Dict],
    quant_candidates_a: Sequence[Dict],
    hidden_dim: int = 768,
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and name != "lm_head":
            mp = MixedPrecisionQATLinearEnhanced(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                hidden_dim=hidden_dim,
                layer_name=full_name,
                quant_candidates_w=quant_candidates_w,
                quant_candidates_a=quant_candidates_a,
            )
            with torch.no_grad():
                mp.weight.copy_(child.weight)
                if child.bias is not None and mp.bias is not None:
                    mp.bias.copy_(child.bias)
            mp.train(child.training)
            setattr(module, name, mp)
        else:
            replace_linears_with_qat(
                child,
                quant_candidates_w=quant_candidates_w,
                quant_candidates_a=quant_candidates_a,
                hidden_dim=hidden_dim,
                prefix=full_name,
            )


def make_collate_fn(tokenizer: AutoTokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = tokenizer.pad(batch, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
    return collate_fn


class StageController:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
    def set_stage(self, stage: str) -> None:
        for m in self.model.modules():
            if not isinstance(m, MixedPrecisionQATLinearEnhanced):
                continue
            if stage == "fp32":
                m.search_w = False
                m.search_a =  False
                for w in m.quantizers_w:
                    w.enable = False
                for a in m.quantizers_a:
                    a.enable = False            
                m.logits_w.requires_grad_(False)
                m.logits_a.requires_grad_(False)            
                for p in m.quantizer_parameters():
                    p.requires_grad_(False)
            elif stage == "weight":
                m.search_w = True
                m.search_a =  False
                for w in m.quantizers_w:
                    w.enable = True
                for a in m.quantizers_a:
                    a.enable = False
                m.logits_w.requires_grad_(True)
                m.logits_a.requires_grad_(False)
                for p in m.quantizer_parameters():
                    p.requires_grad_(True)
            elif stage == "full":
                m.search_w = True
                m.search_a =  True
                for w in m.quantizers_w:
                    w.enable = True
                for a in m.quantizers_a:
                    a.enable = True
                m.logits_w.requires_grad_(True)
                m.logits_a.requires_grad_(True)
                for p in m.quantizer_parameters():
                    p.requires_grad_(True)
            else:
                raise ValueError(f"Unknown stage: {stage}")


def collect_parameter_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    arch_params: List[nn.Parameter] = []
    model_params: List[nn.Parameter] = []
    seen_arch = set()
    seen_model = set()
    for module in model.modules():
        if isinstance(module, MixedPrecisionQATLinearEnhanced):
            for p in module.architecture_parameters():
                if id(p) not in seen_arch:
                    arch_params.append(p)
                    seen_arch.add(id(p))
            for p in [module.weight, module.bias, *module.quantizer_parameters()]:
                if p is not None and id(p) not in seen_model and id(p) not in seen_arch:
                    model_params.append(p)
                    seen_model.add(id(p))
    for p in model.parameters():
        if id(p) not in seen_arch and id(p) not in seen_model:
            model_params.append(p)
            seen_model.add(id(p))
    return model_params, arch_params

def set_requires_grad(params: Iterable[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)

def compute_avg_seq_len(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1).float().mean()

def compute_model_cost(model: nn.Module, avg_seq_len: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), device=avg_seq_len.device)
    for name, module in model.named_modules():
        if not isinstance(module, MixedPrecisionQATLinearEnhanced):
            continue
        total = total + module.projection_bitops(avg_seq_len)
        if name.endswith("q_proj"):
            a_bits = module.expected_activation_bits()
            total = total + 2.0 * (avg_seq_len ** 2) * module.hidden_dim * (a_bits ** 2)
    return total

def compute_full_precision_cost(model: nn.Module, avg_seq_len: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), device=avg_seq_len.device)
    for name, module in model.named_modules():
        if not isinstance(module, MixedPrecisionQATLinearEnhanced):
            continue
        total = total + module.full_precision_projection_bitops(avg_seq_len)
        if name.endswith("q_proj"):
            total = total + 2.0 * (avg_seq_len ** 2) * module.hidden_dim * 32.0 * 32.0
    return total

def compute_error_penalty(model: nn.Module) -> torch.Tensor:
    penalties = [m.last_error_penalty for m in model.modules() if isinstance(m, MixedPrecisionQATLinearEnhanced)]
    if not penalties:
        ref = next(model.parameters())
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.stack(penalties).mean()

def compute_entropy(model: nn.Module) -> torch.Tensor:
    entropies = []
    for m in model.modules():
        if isinstance(m, MixedPrecisionQATLinearEnhanced):
            p_w = F.softmax(m.logits_w, dim=-1)
            p_a = F.softmax(m.logits_a, dim=-1)
            ent_w = -(p_w * (p_w.clamp_min(1e-8).log())).sum()
            ent_a = -(p_a * (p_a.clamp_min(1e-8).log())).sum()
            entropies.append(ent_w + ent_a)
    if not entropies:
        ref = next(model.parameters())
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.stack(entropies).mean()

def extract_selected_quantizers(model: nn.Module) -> Dict[str, Dict[str, Dict[str, object]]]:
    selected: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, MixedPrecisionQATLinearEnhanced):
            continue
        w_idx = int(module.logits_w.argmax().item())
        a_idx = int(module.logits_a.argmax().item())
        selected[name] = {
            "weight": {
                "bits": module.bits_w_list[w_idx],
                "qtype": module.quant_candidates_w[w_idx]["qtype"],
            },
            "activation": {
                "bits": module.bits_a_list[a_idx],
                "qtype": module.quant_candidates_a[a_idx]["qtype"],
            },
        }
    return selected

def train_one_epoch(
    model: nn.Module,
    base_size: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_model: torch.optim.Optimizer,
    optimizer_arch: torch.optim.Optimizer,
    model_params: Sequence[nn.Parameter],
    arch_params: Sequence[nn.Parameter],
    cost_lambda: float,
    error_lambda: float,
    entropy_lambda: float,
    fp_cost: torch.Tensor,
    device: torch.device,
    controller: StageController,
    warmup_fp_steps: int,
    warmup_act_steps: int,
    global_step: int,
    log_every: int,
    save_dir: str,
) -> int:
    model.train()
    val_iter = iter(val_loader)
    for step, batch in enumerate(train_loader):
        global_step += 1         
        if global_step < warmup_fp_steps:
            stage = "fp32"
        elif global_step < warmup_act_steps:
            stage = "weight"
        else:
            stage = "full"
        controller.set_stage(stage)
        train_batch = {k: v.to(device) for k, v in batch.items()}
        set_requires_grad(arch_params, False)
        set_requires_grad(model_params, True)
        controller.set_stage(stage)
        optimizer_model.zero_grad(set_to_none=True)
        train_out = model(
            input_ids=train_batch["input_ids"],
            attention_mask=train_batch["attention_mask"],
            labels=train_batch["labels"],
        )
        train_loss = train_out.loss
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        optimizer_model.step()
        if stage != "fp32":
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            set_requires_grad(model_params, False)
            set_requires_grad(arch_params, True)
            controller.set_stage(stage)
            optimizer_arch.zero_grad(set_to_none=True)
            val_input_ids = val_batch["input_ids"].to(device).long()
            val_labels    = val_batch["labels"].to(device).long()
            val_out = model(
                input_ids=val_batch["input_ids"],
                attention_mask=val_batch["attention_mask"],
                labels=val_batch["labels"],
            )
            linear_cost = torch.tensor(0.0, device=device)
            attention_cost = torch.tensor(0.0, device=device)
            avg_seq_len = val_input_ids.size(1)
    
            for name, module in model.named_modules():
                if not isinstance(module, MixedPrecisionQATLinearEnhanced):
                    continue
    
                if "c_fc" in name or "c_proj" in name:
                    linear_cost += module.expected_linear_cost(avg_seq_len)
    
                elif "attn" in name:
                    linear_cost += module.expected_attention_cost(avg_seq_len)
                    if "q" in name:
                        attention_cost += module.attention_macs_add(avg_seq_len)
    
            normalized_cost = (linear_cost + attention_cost) / base_size
            avg_seq_len = compute_avg_seq_len(val_batch["attention_mask"])
            cost = compute_model_cost(model, avg_seq_len) / fp_cost.clamp_min(1e-12)
            error_penalty = compute_error_penalty(model)
            entropy = compute_entropy(model)
            upper_loss = val_out.loss + cost_lambda * normalized_cost + error_lambda * error_penalty + entropy_lambda * entropy

            active_arch_params = [p for p in arch_params if p.requires_grad]
            
            if active_arch_params:
                arch_grads = torch.autograd.grad(upper_loss, active_arch_params, allow_unused=True)
                for p, g in zip(active_arch_params, arch_grads):
                    p.grad = g
                optimizer_arch.step()
        else:
            val_out = None
            cost = torch.zeros((), device=device)
            error_penalty = torch.zeros((), device=device)
            upper_loss = torch.zeros((), device=device)

        if step % log_every == 0:
            msg = (
                f"step={global_step} stage={stage} train_ce={train_loss.item():.4f} "
                f"cost={cost.item():.4f} error={error_penalty.item():.4f}"
            )
            if val_out is not None:
                msg += f" val_ce={val_out.loss.item():.4f} upper={upper_loss.item():.4f}"
            print(msg)
            log_selected_quantizers(model)
            selected = extract_selected_quantizers(model)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"selected_step_{global_step}.json"), "w", encoding="utf-8") as f:
                json.dump(selected, f, indent=2, ensure_ascii=False)

    return global_step

def log_selected_quantizers(model: nn.Module) -> None:
    selected = extract_selected_quantizers(model)

    print("\nSelected quantizers per layer:")
    for layer, cfg in selected.items():
        w_bits = cfg["weight"]["bits"]
        w_q = cfg["weight"]["qtype"]

        a_bits = cfg["activation"]["bits"]
        a_q = cfg["activation"]["qtype"]

        print(
            f"{layer:40s} | "
            f"W: {w_bits:2d} bit ({w_q}) | "
            f"A: {a_bits:2d} bit ({a_q})"
        )

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLM.from_pretrained(model_name)

    quant_candidates_w = [
        {"bitwidth": 32, "qtype": "identity"},
        {"bitwidth": 4, "qtype": "block_uniform"},
        {"bitwidth": 8, "qtype": "block_uniform"},
        {"bitwidth": 16, "qtype": "block_uniform"},
        {"bitwidth": 4, "qtype": "lsq"},
        {"bitwidth": 8, "qtype": "lsq"},
        {"bitwidth": 16, "qtype": "lsq"},

    ]
    quant_candidates_a = [
        {"bitwidth": 32, "qtype": "identity"},
        {"bitwidth": 4, "qtype": "asym_pact"},
        {"bitwidth": 8, "qtype": "asym_pact"},
        {"bitwidth": 16, "qtype": "asym_pact"},
        {"bitwidth": 4, "qtype": "dynamic_token"},
        {"bitwidth": 8, "qtype": "dynamic_token"},
        {"bitwidth": 16, "qtype": "dynamic_token"},
        {"bitwidth": 4, "qtype": "lsq"},
        {"bitwidth": 8, "qtype": "lsq"},
        {"bitwidth": 16, "qtype": "lsq"},
    ]

    replace_linears_with_qat(model, quant_candidates_w, quant_candidates_a)
    model.to(device)
    dataset_name = "monology/pile-uncopyrighted" 
    ds_train = load_dataset(dataset_name, split="train", streaming=True)
    ds_train = ds_train.take(20000)
    ds_val = load_dataset(dataset_name, split="train", streaming=True)
    ds_val = ds_val.skip(20000).take(2000)
    ds_train = ds_train.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
    ds_val = ds_val.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
    sample_element = next(iter(ds_train))
    all_columns = list(sample_element.keys())
    print(f"Removing columns: {all_columns}")

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=128)
    train_tok = ds_train.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=all_columns
    )
    val_tok = ds_val.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=all_columns
    )
    train_tok = train_tok.with_format("torch")
    val_tok = val_tok.with_format("torch")

    collate = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_tok, batch_size=4, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_tok, batch_size=4, collate_fn=collate, num_workers=2)
    calibration_batch = next(iter(train_loader))
    initialize_lsq_parameters(model, calibration_batch, device)
    model_params, arch_params = collect_parameter_groups(model)
    optimizer_model = AdamW(model_params, lr=2e-5, betas=(0.9, 0.95), weight_decay=0.01)
    optimizer_arch = AdamW(arch_params, lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    controller = StageController(model)
    controller.set_stage("fp32")
    
    sample_batch = next(iter(train_loader))
    avg_seq_len = compute_avg_seq_len(sample_batch["attention_mask"].to(device))
    fp_cost = compute_full_precision_cost(model, avg_seq_len)
    global_step = 0
    save_dir = "best_bits_steps_fixed"
    os.makedirs(save_dir, exist_ok=True)
    pad_id = tokenizer.pad_token_id
    avg_seq_len = compute_avg_seq_lenn(train_loader, pad_id)
    
    max_linear_cost = torch.tensor(0.0, device=device)
    max_attention_macs_add = torch.tensor(0.0, device=device)
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
    for epoch in range(3):
        global_step = train_one_epoch(
            model=model,
            base_size=base_size,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_model=optimizer_model,
            optimizer_arch=optimizer_arch,
            model_params=model_params,
            arch_params=arch_params,
            cost_lambda=0.02,
            error_lambda=0.05,
            entropy_lambda=1e-4,
            fp_cost=fp_cost,
            device=device,
            controller=controller,
            warmup_fp_steps=500,
            warmup_act_steps=1500,
            global_step=global_step,
            log_every=50,
            save_dir=save_dir,
        )
        print(f"finished epoch {epoch + 1}, global_step={global_step}")

    with open(os.path.join(save_dir, "selected_final.json"), "w", encoding="utf-8") as f:
        json.dump(extract_selected_quantizers(model), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
