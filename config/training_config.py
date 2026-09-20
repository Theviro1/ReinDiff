import torch
from config.default_config import *

layer_weight = [
    0.02, 0.01,         # 64*64 down-block
    0.02, 0.01,         # 32*32 down-block
    0.02, 0.25,         # 16*16 down-block
    0.25, 0.02, 0.01,   # 16*16 up-block
    0.02, 0.01, 0.01,   # 32*32 up-block
    0.03, 0.02, 0.01,   # 64*64 up-block
    0.25                # 8*8 mid-block
]
sample_timesteps = list(range(10, 36))

LAYER_WEIGHT = layer_weight
SAMPLE_TIMESTEPS = sample_timesteps
LORA_RANGE = (30, 49)

SFT_CONFIG = {
    'batch_size':4,
    'lr':1e-4,
    'epoches':3,
    'lambda_diff':0.2,
    'lambda_attn':2,
    'lr_decay':0.95,
    'soft_mask':(1.0, 0.0),
    'sample_timesteps':sample_timesteps,
    'layer_weight':layer_weight,
    'dtype':torch.float32,
    'device':'cuda',
    'lora_path':LORA_SFT_PATH,
    'record_folder_path':LOSS_FOLDER_PATH
}

RL_REGULAR_CONFIG = {
    'batch_size':4,
    'lr':1e-4,
    'epoches':5,
    'lr_decay':0.95,
    'eta':1.0,
    'lambda_kl':0,
    'lambda_reward':1,
    'entropy_weight':1,
    'coverage_weight':0.5,
    'overlap_weight':2,
    'sample_timesteps':sample_timesteps,
    'layer_weight':layer_weight,
    'dtype':torch.float32,
    'device':'cuda',
    'lora_path':LORA_RL_REGULAR_PATH,
    'record_folder_path':LOSS_FOLDER_PATH
}


RL_REGULAR_CHAIN_CONFIG = {
    'batch_size':1,
    'lr':1e-4,
    'epoches':2,
    'lr_decay':0.95,
    'eta':1.0,
    'lambda_kl':0.001,
    'entropy_weight':2,
    'coverage_weight':1,
    'overlap_weight':4,
    'sample_timesteps':sample_timesteps,
    'layer_weight':layer_weight,
    'dtype':torch.float32,
    'device':'cuda',
    'lora_path':LORA_RL_REGULAR_CHAIN_PATH,
    'record_folder_path':LOSS_FOLDER_PATH,
}

RL_M2F_CONFIG = {
    'batch_size':4,
    'lr':1e-4,
    'epoches':1,
    'lr_decay':0.95,
    'eta':1.0,
    'lambda_kl':0.01,
    'lambda_reward':3,
    'sample_timesteps':sample_timesteps,
    'layer_weight':layer_weight,
    'dtype':torch.float32,
    'device':'cuda',
    'lora_path':LORA_RL_M2F_PATH,
    'record_folder_path':LOSS_FOLDER_PATH
}

RL_M2F_CHAIN_CONFIG = {
    'batch_size':1,
    'lr':1e-4,
    'lr_decay':0.95,
    'eta':1.0,
    'lambda_kl':0.001,
    'sample_timesteps':sample_timesteps,
    'layer_weight':layer_weight,
    'dtype':torch.float32,
    'device':'cuda',
    'lora_path':LORA_RL_M2F_CHAIN_PATH,
    'record_folder_path':LOSS_FOLDER_PATH
}