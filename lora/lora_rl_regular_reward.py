from sdlib import diffusion_utils, dataset_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.stable_diffusion import StableDiffusion
from lora.peft_model import disable_lora, enable_lora
import random
import torch
from torch.utils.data import DataLoader
import pickle
import os
import math
from typing import Any
from tqdm import tqdm

def minimax(A:torch.Tensor):
    return (A - A.min()) / (A.max() - A.min() + 1e-8)

def softmax(A:torch.Tensor):
    return torch.nn.functional.softmax(A.view(-1), dim=0).view_as(A)

def entropy_score(A:torch.Tensor):
    # input A suppose to be a specific merged attention map for a token, shaped as (64, 64)
    alpha = 0.1
    h, w = A.shape
    A = softmax(A)
    entropy = -(A * A.log()).sum()
    max_entropy = math.log(h * w)
    score = ((1 - entropy / max_entropy)**alpha).item()
    return score

def coverage_score(A:torch.Tensor):
    base_precentage = 0.08
    A = softmax(A)
    threshold = A.mean()
    active_ratio = (A > threshold).float().mean()
    score = min((active_ratio / base_precentage).item(), 1.0)
    return score

def overlap_score(A:torch.Tensor, B:list[torch.Tensor]):
    scores = []
    for b in B:
        assert A.shape == b.shape
        if torch.equal(A, b): 
            continue
        A = minimax(A)
        b = minimax(b)
        overlap = (A * b).sum() / A.numel()
        scores.append((1 - overlap).item())
    length = len(scores)
    score = sum(scores) / length
    return score


def collate_fn(batch):
    prompts = [item['prompt'] for item in batch]  # list[str]
    indexs = [item['idx'] for item in batch]  # list[list[int]]
    return {
        'prompts':prompts,
        'indexs':indexs,
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
        ann_path = data['annotation_path']
        prompt = data['prompt']
        with open(ann_path, 'rb') as f:
            labels = pickle.load(f)
        labels = dataset_utils.remap_index(prompt, labels)
        indexs = [index for index, cat_img in labels]
        return {
            'prompt':prompt,
            'idx':indexs
        }

    def __len__(self):
        return len(self.datas)
    





class DiffusionLoraRLRegularTrainer:
    def __init__(
        self,
        peft_model:StableDiffusion,
        lora_path:str,
        record_folder_path:str,

        epoches:int,
        batch_size:int,
        lr:float,
        lr_decay:float,
        eta:float,
        dtype:torch.dtype,
        device:torch.device,

        lambda_kl:float,
        lambda_reward:float,
        entropy_weight:float,
        overlap_weight:float,
        coverage_weight:float,
        sample_timesteps:list[int],
        layer_weight:list[float],
    ):
        self.peft_model = peft_model
        self.recorder = DiffusionLoraRLLossRecorder(os.path.join(record_folder_path, 'RL_regular_loss.png'))
        # hyperparameters
        self.epoches = epoches
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.eta = eta
        self.dtype = dtype
        self.device = device
        # specific arguments
        self.lora_path = lora_path
        self.entropy_weight = entropy_weight
        self.overlap_weight = overlap_weight
        self.coverage_weight = coverage_weight
        self.lambda_kl = lambda_kl
        self.lambda_reward = lambda_reward
        self.layer_weight = layer_weight
        self.sample_timesteps = sample_timesteps

    
    def frozen(self):
        for name, param in self.peft_model.unet.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    


    
    def reward(self, merged_attn_scores:torch.Tensor, target_indexs:list[list[int]])->torch.Tensor:
        assert merged_attn_scores.ndim == 4  #(batch_size, num_tokens, h, w)
        critic_values = []
        for b, indices in enumerate(target_indexs):
            # indice is a list[int] for all target tokens
            b_score = 0
            token_attn_scores = [merged_attn_scores[b, idx] for idx in indices] # multiple token attn_score for a batch data
            for token_attn_score in token_attn_scores:
                entropy = entropy_score(token_attn_score)
                coverage = coverage_score(token_attn_score)
                overlap = overlap_score(token_attn_score, token_attn_scores)
                token_value = self.entropy_weight * entropy + self.coverage_weight * coverage + self.overlap_weight * overlap
                b_score += token_value / len(indices)
            critic_values.append(b_score)
        return torch.tensor(critic_values).to(dtype=self.dtype, device=self.device)  # (batch_size, )
    
 

    def train(self):
        self.frozen()
        optimizer = torch.optim.Adam([p for p in self.peft_model.unet.parameters() if p.requires_grad], lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        datas = dataset_utils.select()
        dataset = DiffusionLoraRLDataset(datas, self.dtype, self.device)
        loader = DataLoader(dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)
        # acquire s, V(s), a, s', V(s'), R then acquire (A, a)
        cur_batch = 0
        for i in range(self.epoches):
            for batch in tqdm(loader, desc=f'training epoch{i+1}'):
                cur_batch += 1
                prompts = batch['prompts']
                indexs = batch['indexs']
                with torch.no_grad():
                    t_ddim = random.choice(self.sample_timesteps)
                    t = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim)
                    t_prev = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim-1)
                    embeddings = diffusion_utils.encode_prompt(self.peft_model.pipe, prompts)
                    latents = diffusion_utils.latent_t_rev(self.peft_model, embeddings, t_ddim)
                # KL: kl-regularzation
                    disable_lora(self.peft_model)
                    noise_preds_kl = diffusion_utils.predict(self.peft_model.unet, latents, t, embeddings)
                    enable_lora(self.peft_model)
                # a:logprobs
                noise_preds = diffusion_utils.predict(self.peft_model.unet, latents, t, embeddings)
                _, logprobs = diffusion_utils.step_logprob(self.peft_model.scheduler, latents, t, noise_preds, t_prev, self.eta)
                kl_loss = torch.nn.functional.mse_loss(noise_preds, noise_preds_kl)
                del latents
                torch.cuda.empty_cache()
                with torch.no_grad():
                # R:rewards
                    merged_attn_scores = self.peft_model.merged_attn_scores
                    rewards = self.reward(merged_attn_scores, indexs)
                    del merged_attn_scores
                    torch.cuda.empty_cache()
                loss = (- self.lambda_reward * rewards * logprobs).sum() + self.lambda_kl * kl_loss
                # step
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
    


    def unroll_chain_train(self, unroll_chain_length:int=3):
        self.frozen()
        optimizer = torch.optim.Adam([p for p in self.peft_model.unet.parameters() if p.requires_grad], lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        datas = dataset_utils.select()
        dataset = DiffusionLoraRLDataset(datas, self.dtype, self.device)
        loader = DataLoader(dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)
        # acquire s, V(s), a, s', V(s'), R then acquire (A, a)
        cur_batch = 0
        for i in range(self.epoches):
            for batch in tqdm(loader, desc=f'training epoch{i+1}'):
                cur_batch += 1
                prompts = batch['prompts']
                indexs = batch['indexs']
                with torch.no_grad():
                    t_ddim = random.choice(self.sample_timesteps)
                    embeddings = diffusion_utils.encode_prompt(self.peft_model.pipe, prompts)
                    latents = diffusion_utils.latent_t_rev(self.peft_model, embeddings, t_ddim)
                loss = 0
                for t_ddim_sample in range(t_ddim, t_ddim-unroll_chain_length, -1):
                    with torch.no_grad():
                    # KL: kl-regularzation
                        t = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim_sample)
                        t_prev = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim_sample-1)
                        disable_lora(self.peft_model)
                        noise_preds_kl = diffusion_utils.predict(self.peft_model.unet, latents, t, embeddings)
                        enable_lora(self.peft_model)
                    # a:logprobs
                    noise_preds = diffusion_utils.predict(self.peft_model.unet, latents, t, embeddings)
                    next_latents, logprobs = diffusion_utils.step_logprob(self.peft_model.scheduler, latents, t, noise_preds, t_prev, self.eta)
                    kl_loss = torch.nn.functional.mse_loss(noise_preds, noise_preds_kl)
                    loss += self.lambda_kl * kl_loss
                    # R:rewards
                    with torch.no_grad():
                        merged_attn_scores = self.peft_model.merged_attn_scores
                        rewards = self.reward(merged_attn_scores, indexs)
                        del merged_attn_scores
                        torch.cuda.empty_cache()
                    loss += (- rewards * logprobs).sum()
                    latents = next_latents
                # step
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


        