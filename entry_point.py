import apex
import numpy as np
import torch
from torch import nn

from model import GPTConfig, GPT

# STEP 1: Manipulate batch size.
batch_size = 12 # 12 -> 96
block_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model
n_layer = 8 
n_head = 8 
n_embd = 256 
dropout = 0.0 
vocab_size = 65 
bias = False 
enable_flash_attention = False

# adamw optimizer
learning_rate = 6e-4 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# optimizer
def configure_optimizer(model, weight_decay, learning_rate, betas):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    return optimizer

# optimizer apex
def configure_optimizer_apex(model, weight_decay, learning_rate, betas):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = apex.optimizers.FusedAdam(params=optim_groups,lr=learning_rate,betas=betas,adam_w_mode=True)

    return optimizer

def deepview_model_provider():
    # model init
    # ---------------------------------------------
    # STEP 3: Enable flash attention
    #enable_flash_attention = True
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, enable_flash_attention=enable_flash_attention) 
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    return model.to(device) 

def deepview_input_provider(batch_size=batch_size):
    data = np.random.randint(vocab_size, size=(batch_size,block_size+1))
    x = torch.stack([torch.from_numpy((data[i,:-1]).astype(np.int64)) for i in range(batch_size)])
    y = torch.stack([torch.from_numpy((data[i,1:]).astype(np.int64)) for i in range(batch_size)])
    
    return (x.to(device), y.to(device))

def deepview_iteration_provider(model):
    criterion = nn.CrossEntropyLoss()
    # ---------------------------------------------
    # STEP 4: Fused optimizer
    #optimizer = configure_optimizer_apex(model,weight_decay, learning_rate, (beta1, beta2))
    optimizer = configure_optimizer(model,weight_decay, learning_rate, (beta1, beta2))

    def iteration(inputs, targets):
        optimizer.zero_grad()
        # ---------------------------------------------
        # STEP 2: Automatic Mixed Precision
        #with torch.autocast(device_type=device):
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    return iteration
