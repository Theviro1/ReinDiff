from sdlib import diffusion_utils, dataset_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.stable_diffusion import StableDiffusion
from model.clip import CLIP
from config.default_config import LAYER_WEIGHT, SAMPLE_TIMESTEPS_RL_REGULAR
from lora.peft_model import disable_lora, enable_lora
import random
import torch
from torch.utils.data import DataLoader
import os
from typing import Any
from tqdm import tqdm


def collate_fn(batch):
    prompts = [item['prompt'] for item in batch]  # list[str]
    return {
        'prompts':prompts,
    }

class DiffusionLoraRLLossRecorder:
    def __init__(self, save_path:str, save_gap:int=50):
        self.losses = []
        self.steps = []
        self.save_path = save_path
        self.save_gap = save_gap  # every `save_gap` batch refresh image

    def update(self, loss:float, step:int):
        if step % self.save_gap == 0:
            self.losses.append(loss)
            self.steps.append(step)
            self.save_plot()

    def save_plot(self):
        plt.figure()
        plt.plot(self.steps, self.losses, marker='o', linestyle='-', color='b', label='loss')
        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.title("Reinforcement Learning Batch Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path, dpi=300)
        plt.close()



class DiffusionLoraRLDataset(torch.utils.data.Dataset):
    def __init__(self, datas:list[dict[str, Any]], dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        self.datas = datas
    
    def __getitem__(self, index):
        data = self.datas[index]
        prompt = data['prompt']
        return {
            'prompt':prompt,
        }

    def __len__(self):
        return len(self.datas)
    





class DiffusionLoraRLCLIPTrainer:
    def __init__(
        self,
        peft_model:StableDiffusion,
        clip:CLIP,
        lora_path:str,
        folder_path:str,

        epoches:int=3,
        batch_size:int=4,
        lr:float=1e-4,
        lr_decay:float=0.95,
        gamma:float=0.95,
        eta:float=1.0,

        lambda_reward:float=5.0,
        lambda_kl:float=0.2,
        sample_timesteps:list[int]=SAMPLE_TIMESTEPS_RL_REGULAR,
        layer_weight:list[float]=LAYER_WEIGHT,
    ):
        self.peft_model = peft_model
        self.clip = clip
        self.recorder = DiffusionLoraRLLossRecorder(os.path.join(folder_path, 'RL_clip_loss.png'))
        # hyperparameters
        self.epoches = epoches
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.eta = eta
        self.dtype = torch.float32
        self.device = 'cuda'
        # specific arguments
        self.lora_path = lora_path
        self.lambda_kl = lambda_kl
        self.lambda_reward = lambda_reward
        self.layer_weight = layer_weight
        self.sample_timesteps = sample_timesteps

    
    def frozen(self):
        for name, param in self.peft_model.unet.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    


    
    def reward(self, prompts:list[str], latents:torch.Tensor, t:int, noise_preds:torch.Tensor)->torch.Tensor:
        origin_latents = diffusion_utils.origin(self.peft_model.scheduler, latents, t, noise_preds)
        origin_images = diffusion_utils.decode_latent(self.peft_model.vae, origin_latents)
        origin_images = diffusion_utils.reformat_image(origin_images)
        rewards = self.clip(prompts, origin_images)
        return rewards  # (batch_size, )
    
 

    def train(self):
        self.frozen()
        optimizer = torch.optim.Adam([p for p in self.peft_model.unet.parameters() if p.requires_grad], lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        datas = dataset_utils.select()
        dataset = DiffusionLoraRLDataset(datas, self.dtype, self.device)
        loader = DataLoader(dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)
        # acquire R for each s
        cur_batch = 0
        for i in range(self.epoches):
            for batch in tqdm(loader, desc=f'training epoch{i+1}'):
                cur_batch += 1
                prompts = batch['prompts']
                with torch.no_grad():
                    t_ddim = random.choice(self.sample_timesteps)
                    t = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim)
                    t_prev = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim-1)
                    embeddings = diffusion_utils.encode_prompt(self.peft_model.pipe, prompts)
                    latents = diffusion_utils.latent_t_comb(self.peft_model, embeddings, t_ddim)
                # KL: kl-regularzation
                    disable_lora(self.peft_model)
                    noise_preds_kl = diffusion_utils.predict(self.peft_model.unet, latents, t, embeddings)
                    enable_lora(self.peft_model)
                # a:logprobs
                noise_preds = diffusion_utils.predict(self.peft_model.unet, latents, t, embeddings)
                _, logprobs = diffusion_utils.step_logprob(self.peft_model.scheduler, latents, t, noise_preds, t_prev, self.eta)
                kl_loss = torch.nn.functional.mse_loss(noise_preds, noise_preds_kl)
                with torch.no_grad():
                # R:rewards
                    rewards = self.reward(prompts, latents, t, noise_preds)
                torch.cuda.empty_cache()
                # step
                loss = (- rewards * logprobs).sum() + self.lambda_kl * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # finish a batch
                self.recorder.update(loss.item(), cur_batch)
            # finish an epoch
            lr_scheduler.step()
        # finish training
        lora_state_dict = {k:v for k, v in self.peft_model.unet.state_dict().items() if 'lora_' in k.lower()}
        torch.save(lora_state_dict, self.lora_path)
        self.peft_model.unet.eval()