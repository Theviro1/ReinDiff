import torch
import math
from model.stable_diffusion import StableDiffusion


class LoRALinear(torch.nn.Module):
    def __init__(self, original:torch.nn.Linear, r:int, alpha:int, dropout:float=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original = original
        std_dev = 1 / math.sqrt(r)
        self.lora_a = torch.nn.Parameter(torch.randn(original.in_features, r) * std_dev)
        self.lora_b = torch.nn.Parameter(torch.zeros(r, original.out_features))
        self.lora_dropout = torch.nn.Dropout(dropout)
        self.scaling = alpha / r
        self.saved_scaling = alpha / r
    
    def forward(self, x:torch.Tensor):
        return self.original(x) + self.scaling * (self.lora_dropout(x) @ self.lora_a @ self.lora_b)

def get_peft_model(model:StableDiffusion, lora_config:dict=None):
    unet = model.unet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    if lora_config is not None:
        lora_rank = lora_config['lora_rank']
        lora_alpha = lora_config['lora_alpha']
    else:
        lora_rank = 16
        lora_alpha = 4

    # Downblocks
    for i, downblock in enumerate(unet.down_blocks):
        if hasattr(downblock, 'attentions'):
            for attn_module in downblock.attentions:
                for trfms_module in attn_module.transformer_blocks:
                    # list of BasicTransformerBlock
                    # self-attention
                    trfms_module.attn1.to_q = LoRALinear(trfms_module.attn1.to_q, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn1.to_k = LoRALinear(trfms_module.attn1.to_k, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn1.to_v = LoRALinear(trfms_module.attn1.to_v, lora_rank, lora_alpha).to(device, dtype)
                    # cross-attention
                    trfms_module.attn2.to_q = LoRALinear(trfms_module.attn2.to_q, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn2.to_k = LoRALinear(trfms_module.attn2.to_k, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn2.to_v = LoRALinear(trfms_module.attn2.to_v, lora_rank, lora_alpha).to(device, dtype)
    # Midblocks
    if hasattr(unet.mid_block, 'attentions'):
        for attn_module in unet.mid_block.attentions:
            for trfms_module in attn_module.transformer_blocks:
                # list of BasicTransformerBlock
                # self-attention
                trfms_module.attn1.to_q = LoRALinear(trfms_module.attn1.to_q, lora_rank, lora_alpha).to(device, dtype)
                trfms_module.attn1.to_k = LoRALinear(trfms_module.attn1.to_k, lora_rank, lora_alpha).to(device, dtype)
                trfms_module.attn1.to_v = LoRALinear(trfms_module.attn1.to_v, lora_rank, lora_alpha).to(device, dtype)
                # cross-attention
                trfms_module.attn2.to_q = LoRALinear(trfms_module.attn2.to_q, lora_rank, lora_alpha).to(device, dtype)
                trfms_module.attn2.to_k = LoRALinear(trfms_module.attn2.to_k, lora_rank, lora_alpha).to(device, dtype)
                trfms_module.attn2.to_v = LoRALinear(trfms_module.attn2.to_v, lora_rank, lora_alpha).to(device, dtype)
    # Upblocks
    for i, upblock in enumerate(unet.up_blocks):
        if hasattr(upblock, 'attentions'):
            for attn_module in upblock.attentions:
                for trfms_module in attn_module.transformer_blocks:
                    # list of BasicTransformerBlock
                    # self-attention
                    trfms_module.attn1.to_q = LoRALinear(trfms_module.attn1.to_q, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn1.to_k = LoRALinear(trfms_module.attn1.to_k, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn1.to_v = LoRALinear(trfms_module.attn1.to_v, lora_rank, lora_alpha).to(device, dtype)
                    # cross-attention
                    trfms_module.attn2.to_q = LoRALinear(trfms_module.attn2.to_q, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn2.to_k = LoRALinear(trfms_module.attn2.to_k, lora_rank, lora_alpha).to(device, dtype)
                    trfms_module.attn2.to_v = LoRALinear(trfms_module.attn2.to_v, lora_rank, lora_alpha).to(device, dtype)
    return model


def load_lora_weight(model:StableDiffusion, lora_path:str):
    peft_model_loaded = False
    for name, param in model.unet.named_parameters():
        if 'lora_' in name:
            peft_model_loaded = True
            break
    if not peft_model_loaded:
        raise ValueError('peft model not loaded, use `get_peft_model` first')
    lora_state_dict = torch.load(lora_path, weights_only=True)
    model.unet.load_state_dict(lora_state_dict, strict=False)

def disable_lora(model:StableDiffusion):
    for name, module in model.unet.named_modules():
        if isinstance(module, LoRALinear):
            module.scaling = 0

def enable_lora(model:StableDiffusion):
    for name, module in model.unet.named_modules():
        if isinstance(module, LoRALinear):
            module.scaling = module.saved_scaling

def check_lora_grad(model:StableDiffusion):
        for name, param in model.unet.named_parameters():
            if 'lora_' in name:
                print(f'grad of {name}:')
                print(param.requires_grad)
                print(param.grad.max().item(), param.grad.min().item())
                print('exist nan') if torch.isnan(param.grad).any() else None
    
def check_lora_nan(model:StableDiffusion):
    for name, param in model.unet.named_parameters():
        if 'lora_' in name and torch.isnan(param).any():
            print(name)