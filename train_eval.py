import math
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
    get_cosine_schedule_with_warmup,
)
from quantizers import (
    IdentityQuantizer,
    BlockUniformWeightQuantizer,
    NF4WeightQuantizer,
    AsymmetricPACT,
    DynamicTokenQuantizer,
    LSQPlusWeightQuantizer,
    LSQPlusActQuantizer,
)

MODEL_NAME = "EleutherAI/gpt-neo-125M"
SEQ_LEN = 512
BATCH_SIZE = 4
LOG_INTERVAL = 50
TOTAL_STEPS = 4000
ACCUM_STEPS = 8
WEIGHT_QUANT_START = 500
ACT_QUANT_START = 1500
LR_WEIGHTS = 5e-6
LR_QUANT = 1e-4
LOG_INTERVAL = 50 
VAL_INTERVAL = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = DEVICE

def build_quantizer(config_dict, kind='weight'):
    bits = config_dict.get("bits", 32)
    qtype = config_dict.get("qtype", "symmetric")
    if bits >= 32:
        return IdentityQuantizer(bitwidth=32)
    if kind == "weight":
        if qtype == "uniform":
            return BlockUniformWeightQuantizer(
                bitwidth=bits, # обычно веса делаем per-channel
            )
        elif qtype == "lsq":
            return LSQPlusWeightQuantizer(
                num_bits=bits, # обычно веса делаем per-channel
            )
        elif qtype == "nf4":
            return NF4WeightQuantizer(
                )
        else:
            raise ValueError(f"Unknown weight qtype: {qtype}")

    elif kind == "activation":
        if qtype == "pact":
            return AsymmetricPACT(
                bits=bits,
            )
        if qtype == "dynamic":
            return DynamicTokenQuantizer(
                bitwidth=bits,
            )
        if qtype == "lsq":
            return LSQPlusActQuantizer(
                num_bits=bits,
            )
        else:
            raise ValueError(f"Unknown activation qtype: {qtype}")

class QATLinear(nn.Module):
    def __init__(self, linear_layer, w_conf, a_conf):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = nn.Parameter(linear_layer.weight.detach().clone())
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.detach().clone())
        else:
            self.register_parameter('bias', None)
        self.wq = build_quantizer(w_conf, kind='weight')
        self.aq = build_quantizer(a_conf, kind='activation')

    def forward(self, x):
        x_q = self.aq(x)
        w_q = self.wq(self.weight)        
        return F.linear(x_q, w_q, self.bias)

def replace_layers_with_qat(model, selected_bits):
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if name in selected_bits and isinstance(module, nn.Linear):
            conf = selected_bits[name]
            w_conf = conf.get("weight", {"bits": 32})
            a_conf = conf.get("activation", {"bits": 32})
            qat_layer = QATLinear(module, w_conf, a_conf)
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                child_name = name
                parent = model
                
            setattr(parent, child_name, qat_layer)
            replaced_count += 1
def set_quant_mode(model, weight=False, act=False):
    for m in model.modules():
        if hasattr(m, 'enable'):
            if isinstance(m, (
                
                BlockUniformWeightQuantizer,  
            )):
                m.enable = weight
            if isinstance(m, (
                AsymmetricPACT,
            )):
                m.enable = act

@torch.no_grad()
def evaluate_ppl(model, loader):
    model.eval()
    t_loss, t_toks = 0.0, 0
    for i, batch in enumerate(loader):
        x = batch["input_ids"].to(DEVICE)
        out = model(input_ids=x, labels=x)
        t_loss += out.loss.item() * x.numel()
        t_toks += x.numel()
        if i >= 10: break
    model.train()
    return math.exp(t_loss / t_toks) if t_toks > 0 else 0

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
try:
    dataset_stream = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    dataset_stream = dataset_stream.shuffle(seed=42, buffer_size=10000)
    val_data = list(dataset_stream.take(1000))
    ds_val = Dataset.from_list(val_data)
    ds_train = dataset_stream.skip(1000)

@torch.no_grad()
def initialize_lsq_parameters(model: nn.Module, init_batch: Dict[str, torch.Tensor], device: torch.device):
    model.eval()
    inputs = init_batch["input_ids"].to(device)
    attention_mask = init_batch["attention_mask"].to(device)
    for m in model.modules():
        if isinstance(m, QATLinear):
            m.search_w = False
            m.search_a = False
            for q in m.quantizers_w: q.enable = True
            for q in m.quantizers_a: q.enable = True
            for q in list(m.quantizers_w) + list(m.quantizers_a):
                if hasattr(q, 'init_done'):
                    q.init_done.fill_(False)
    model(input_ids=inputs, attention_mask=attention_mask)
    model.train()

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN)

tokenized_train = ds_train.map(tokenize_function, batched=True, remove_columns=["text", "meta"])
tokenized_val = ds_val.map(tokenize_function, batched=True, remove_columns=["text", "meta"])

def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length < SEQ_LEN:
        return {k: [] for k in concatenated.keys()}

    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    result = {
        k: [t[i: i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated.items()
    }
    return result


lm_train = tokenized_train.map(group_texts, batched=True)
lm_val = tokenized_val.map(group_texts, batched=True)

collate = lambda b: {
    "input_ids": torch.tensor([x["input_ids"] for x in b], dtype=torch.long),
    "attention_mask": torch.ones(
        (len(b), len(b[0]["input_ids"])), dtype=torch.long
    )
}

train_loader = DataLoader(lm_train, batch_size=BATCH_SIZE, collate_fn=collate)
val_loader = DataLoader(lm_val, batch_size=BATCH_SIZE, collate_fn=collate)
calibration_batch = next(iter(train_loader))
initialize_lsq_parameters(model, calibration_batch, device)
if 'selected_bits' not in globals() or not selected_bits:
    selected_bits_local = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            selected_bits_local[n] = {
                "weight": {"bits": 4, "qtype": "nf4"},
                "activation": {"bits": 8, "qtype": "lsq"}
            }
    replace_layers_with_qat(model, selected_bits_local)
else:
    replace_layers_with_qat(model, selected_bits)

model.to(DEVICE)
model.train()

set_quant_mode(model, weight=True, act=True)

with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i >= 30:
            break
        x = batch["input_ids"].to(DEVICE)
        model(input_ids=x, labels=x)
        
set_quant_mode(model, weight=False, act=False)
w_params, q_params = [], []

for n, p in model.named_parameters():
    if p.requires_grad:
        if "s" in n or "alpha" in n or "beta" in n or "log_scale" in n:
            q_params.append(p)
        else:
            w_params.append(p)

optimizer = torch.optim.AdamW([
    {"params": w_params, "lr": LR_WEIGHTS},
    {"params": q_params, "lr": LR_QUANT}
])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    200,
    TOTAL_STEPS // ACCUM_STEPS
)

curr_step = 0
train_loss_accum = 0.0
steps_in_log = 0

data_iter = iter(train_loader)

while curr_step < TOTAL_STEPS:

    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    if curr_step == WEIGHT_QUANT_START:
        set_quant_mode(model, weight=True, act=False)

    if curr_step == ACT_QUANT_START:
        set_quant_mode(model, weight=True, act=True)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, "is_lsq_param") and p.is_lsq_param:
                    optimizer.state.pop(p, None)

    x = batch["input_ids"].to(DEVICE)
    outputs = model(input_ids=x, labels=x)
    loss = outputs.loss / ACCUM_STEPS
    loss.backward()
    train_loss_accum += outputs.loss.item()
    steps_in_log += 1

    if (curr_step + 1) % ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if (curr_step + 1) % LOG_INTERVAL == 0:
            avg_train_loss = train_loss_accum / steps_in_log
            train_ppl = math.exp(avg_train_loss)

            print(
                f"Step {curr_step+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train PPL: {train_ppl:.2f}"
            )

            train_loss_accum = 0.0
            steps_in_log = 0

        if (curr_step + 1) % VAL_INTERVAL == 0:
            val_ppl = evaluate_ppl(model, val_loader)
            print(f"Step {curr_step+1} | Validation PPL: {val_ppl:.2f}")

    curr_step += 1
