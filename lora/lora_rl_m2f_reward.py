from sdlib import diffusion_utils, dataset_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.stable_diffusion import StableDiffusion
from model.mask2former import Mask2Former
from lora.peft_model import disable_lora, enable_lora
import random
import torch
import pickle
from torch.utils.data import DataLoader
import os
from typing import Any
from tqdm import tqdm

def jaccard_loss(preds:torch.Tensor, target:torch.Tensor):
    # mixed loss with jaccard and BCE
    eps = 1e-6
    scaling_factor = 10  # means preds will mostly be clamped in between
    # both target and prediction shaped like (64, 64) or (batch_size, 64, 64)
    assert preds.shape == target.shape, 'predition shaped different to target'
    # scaling to sigmoid sensitive area [-scaling_factor, scaling_factor]
    with torch.no_grad():
        max_val = preds.abs().max()
        T = max(max_val / scaling_factor, 1.0)
    preds = torch.sigmoid(preds / T)

    intersection = (preds * target).sum(dim=(-2, -1))
    union = (preds + target - preds*target).sum(dim=(-2, -1))
    iou = (intersection + eps) / (union + eps)
    loss = (1 - iou).mean()
    return loss

def collate_fn(batch):
    prompts = [item['prompt'] for item in batch]  # list[str]
    labels = [item['label'] for item in batch]
    return {
        'prompts':prompts,
        'labels':labels
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
        ann_path = data['annotation_path']
        with open(ann_path, 'rb') as f:
            labels = pickle.load(f)
        indexs = [index for index, _ in labels]
        cat_ids = dataset_utils.get_cat_ids(prompt, indexs)
        labels = [(indexs[i], cat_ids[i]) for i in range(len(indexs))]
        labels = dataset_utils.remap_index(prompt, labels)
        return {
            'prompt':prompt,
            'label':labels
        }

    def __len__(self):
        return len(self.datas)
    





class DiffusionLoraRLM2FTrainer:
    def __init__(
        self,
        peft_model:StableDiffusion,
        mask2former:Mask2Former,
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
        sample_timesteps:list[int],
        layer_weight:list[float],
    ):
        self.peft_model = peft_model
        self.mask2former = mask2former
        self.recorder = DiffusionLoraRLLossRecorder(os.path.join(record_folder_path, 'RL_m2f_loss.png'))
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
        self.lambda_kl = lambda_kl
        self.lambda_reward = lambda_reward
        self.layer_weight = layer_weight
        self.sample_timesteps = sample_timesteps

    
    def frozen(self):
        for name, param in self.peft_model.unet.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    


    
    def reward(self, merged_attn_scores:torch.Tensor, labels:list[list[tuple[int, int]]], latents:torch.Tensor, t:int, noise_preds:torch.Tensor)->torch.Tensor:
        # merged_attn_scores shaped as (batch_size, 77, 64, 64)
        # labels shaped as [batch_size, num_target_tokens, (index_in_prompt, cat_id)]
        # segment image
        origin_latents = diffusion_utils.origin(self.peft_model.scheduler, latents, t, noise_preds)
        origin_images = diffusion_utils.decode_latent(self.peft_model.vae, origin_latents)
        seg_results = self.mask2former(origin_images)  # (batch_size, 512, 512)
        seg_results = [dataset_utils.sep_mask(mask) for mask in seg_results]
        batch_size = len(labels)
        mious = []
        for b in range(batch_size):
            seg_result = seg_results[b]  # list[(cat_id, cat_img)]
            label = labels[b]  # list[(idx, cat_id)]
            merged_attn_score = merged_attn_scores[b]   # (77, 64, 64)
            b_iou = []
            for idx, cat_id in label:
                token_attn_score = merged_attn_score[idx]
                cat_img = [img for id, img in seg_result if id == cat_id]  # let id of prompt token equals a sep_mask's id
                if len(cat_img) > 0:
                    cat_img = diffusion_utils.resample(cat_img[0].float())
                else:
                    # skip this idx, cat_id of prompt token if it's not in mask2former's output classes
                    continue
                # acquired (token_attn_score, cat_img) pair, calculate mIou score
                iou = jaccard_loss(token_attn_score, cat_img)
                b_iou.append(iou.item())
            b_miou = sum(b_iou) / len(b_iou) if len(b_iou) > 0 else 0  # early timesteps like 40-20 might get blurry origin images, mask2former might recognize all pixels as background
            mious.append(b_miou)
        return torch.tensor(mious, dtype=self.dtype, device=self.device)  # (batch_size, )
    
 

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
                labels = batch['labels']  # batch_size list[list[(index, cat_id)]]
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
                with torch.no_grad():
                # R:rewards
                    merged_attn_scores = self.peft_model.merged_attn_scores
                    rewards = self.reward(merged_attn_scores, labels, latents, t, noise_preds)
                    del merged_attn_scores, latents
                    torch.cuda.empty_cache()
                # step
                loss = (- self.lambda_reward * rewards * logprobs).sum() + self.lambda_kl * kl_loss
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
        # acquire R for each s
        cur_batch = 0
        for i in range(self.epoches):
            for batch in tqdm(loader, desc=f'training epoch{i+1}'):
                cur_batch += 1
                prompts = batch['prompts']
                labels = batch['labels']  # batch_size list[list[(index, cat_id)]]
                with torch.no_grad():
                    t_ddim = random.choice(self.sample_timesteps)
                    embeddings = diffusion_utils.encode_prompt(self.peft_model.pipe, prompts)
                    latents = diffusion_utils.latent_t_rev(self.peft_model, embeddings, t_ddim)
                loss = 0
                for t_ddim_sample in range(t_ddim, t_ddim-unroll_chain_length, -1):
                    with torch.no_grad():
                        t = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim_sample)
                        t_prev = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim_sample-1)
                    # KL: kl-regularzation
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
                        rewards = self.reward(merged_attn_scores, labels, latents, t, noise_preds)
                        del merged_attn_scores, latents
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



        