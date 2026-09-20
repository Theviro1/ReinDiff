from sdlib import diffusion_utils, dataset_utils
from lora.peft_model import disable_lora, enable_lora
from config.training_config import *
from model.stable_diffusion import StableDiffusion
from typing import Any, Literal
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle
import os
import random
from tqdm import tqdm


def jaccard_loss(preds:torch.Tensor, target:torch.Tensor):
    # mixed loss with jaccard and BCE
    eps = 1e-6
    scaling_factor = 10 # means preds will mostly be clamped in between
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


def dice_loss(preds: torch.Tensor, target: torch.Tensor):
    eps = 1e-6
    scaling_factor = 10  # similar idea as jaccard_loss
    assert preds.shape == target.shape, 'prediction shape different from target'
    # scaling to sigmoid sensitive area [-scaling_factor, scaling_factor]
    with torch.no_grad():
        max_val = preds.abs().max()
        T = max(max_val / scaling_factor, 1.0)
    preds = torch.sigmoid(preds / T)
    # Flatten per sample: supports both (B, H, W) or (H, W)
    intersection = (preds * target).sum(dim=(-2, -1))
    union = preds.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * intersection + eps) / (union + eps)
    loss = (1 - dice).mean()
    return loss

def bce_loss(preds: torch.Tensor, target: torch.Tensor):
    scaling_factor = 10  # similar idea as jaccard_loss
    assert preds.shape == target.shape, 'prediction shape different from target'
    # scaling to sigmoid sensitive area [-scaling_factor, scaling_factor]
    with torch.no_grad():
        max_val = preds.abs().max()
        T = max(max_val / scaling_factor, 1.0)
    preds = preds / T
    loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, target)
    return loss


def collate_fn(batch):
    prompts = [item['prompt'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    labels = [item['labels'] for item in batch]
    return {
        'prompts':prompts,
        'images':images,
        'labels':labels
    }

class DiffusionLoraSFTLossRecorder:
    def __init__(self, save_path:str, save_gap:int=50):
        self.losses = []
        self.steps = []
        self.save_path = save_path
        self.save_gap = save_gap
    
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
        plt.title("Supervised Finetune Learning Batch Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path, dpi=300)
        plt.close()


class DiffusionLoraSFTDataset(torch.utils.data.Dataset):
    def __init__(self, datas:list[dict[str, Any]], soft_mask:tuple[float, float], dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.datas = datas
        self.soft_mask = soft_mask
    
    def __getitem__(self, index):
        data = self.datas[index]
        prompt = data['prompt']
        image = diffusion_utils.read_image(data['image_path'], 'Tensor').to(device=self.device)
        with open(data['annotation_path'], 'rb') as f:
            labels = pickle.load(f)  # labels is a list of tuple like (token_index, seg_image)
        labels = [(index, diffusion_utils.resample(torch.where(mask == 1, self.soft_mask[0], self.soft_mask[1]).to(self.dtype))) for index, mask in labels]  # use soft_mask&resize to (64, 64)
        labels = dataset_utils.remap_index(prompt, labels)
        return {
            'prompt':prompt,
            'image':image,
            'labels':labels
        }

    def __len__(self):
        return len(self.datas)


class DiffusionLoraSFTTrainer:
    def __init__(
        self, 
        peft_model:StableDiffusion,
        lora_path:str,
        record_folder_path:str,

        dtype:torch.dtype,
        device:torch.device,
        batch_size:int, 
        lr:float, 
        epoches:int,
        layer_weight:list,
        soft_mask:tuple[float, float],
        sample_timesteps:list,
        lambda_diff:float,
        lambda_attn:float,
        lr_decay:float,
    ):
        self.peft_model = peft_model
        self.diff_recorder = DiffusionLoraSFTLossRecorder(os.path.join(record_folder_path, 'diff_loss.png'))
        self.attn_recorder = DiffusionLoraSFTLossRecorder(os.path.join(record_folder_path, 'attn_loss.png'))
        # hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.epoches = epoches
        self.device = 'cuda'
        self.dtype = dtype
        # specific training arguments
        self.layer_weight = layer_weight
        self.lambda_attn = lambda_attn
        self.lambda_diff = lambda_diff
        self.soft_mask = soft_mask
        self.sample_timesteps = sample_timesteps
        self.lr_decay = lr_decay
        self.iou_loss_weight = None
        self.bce_loss_weight = None
        # path
        self.lora_path = lora_path
    
    def frozen(self):
        for name, param in self.peft_model.unet.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    
    def schedule_weight(self, epoch:int):
        scheduler = {
            0:(0.3, 0.7),
            1:(0.5, 0.5),
            2:(0.6, 0.4),
        }
        if epoch not in scheduler.keys():
            self.iou_loss_weight, self.bce_loss_weight = (0.7, 0.3)
        else:
            self.iou_loss_weight, self.bce_loss_weight = scheduler[epoch]

    
    def train(self):
        # preprocess
        self.frozen()
        optimizer = torch.optim.Adam(self.peft_model.unet.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        datas = dataset_utils.select()
        dataset = DiffusionLoraSFTDataset(datas, self.soft_mask, self.dtype, self.device)
        loader = DataLoader(dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)
        # train
        cur_batch = 0
        for epoch in range(self.epoches):
            self.schedule_weight(epoch)
            for batch in tqdm(loader, desc=f'training epoch{epoch}'):
                cur_batch += 1
                prompts = batch['prompts']
                images = batch['images']
                labels = batch['labels']
                with torch.no_grad():
                    t_ddim = random.choice(self.sample_timesteps)   # randomly choose a timestep for training
                    t = diffusion_utils.calc_t(self.peft_model.scheduler, t_ddim)
                    latents = diffusion_utils.encode_image(self.peft_model.vae, images)
                    _, noise_latents = diffusion_utils.add_noise(self.peft_model.scheduler, latents, t)
                    embeddings = diffusion_utils.encode_prompt(self.peft_model.pipe, prompts)
                    embeddings = dataset_utils.avg_embeddings(prompts, embeddings)
                    disable_lora(self.peft_model)
                    noise_preds_kl = diffusion_utils.predict(self.peft_model.unet, noise_latents, t, embeddings)
                    enable_lora(self.peft_model)
                    del latents
                    torch.cuda.empty_cache()
                # forward
                noise_preds = diffusion_utils.predict(self.peft_model.unet, noise_latents, t, embeddings)
                merged_attn_scores = self.peft_model.merged_attn_scores  # (batch_size, 77, 64, 64)
                attn_losses, attn_weights = [], []
                # calc_loss
                for b, label in enumerate(labels):
                    for index, mask in label:
                        mask = mask.to(self.device)
                        merged_attn_score = merged_attn_scores[b, index]
                        loss = self.iou_loss_weight*jaccard_loss(merged_attn_score, mask) + self.bce_loss_weight*bce_loss(merged_attn_score, mask)
                        weight = (1 / ((mask == self.soft_mask[0]).sum().sqrt() + 1e-8)).detach()
                        attn_losses.append(loss)
                        attn_weights.append(weight)
                attn_weights = torch.stack(attn_weights)
                attn_weights = attn_weights / attn_weights.sum()
                # sum_loss
                diff_loss = torch.nn.functional.mse_loss(noise_preds, noise_preds_kl)
                attn_loss = (torch.stack(attn_losses)*attn_weights).sum()
                # loss = self.lambda_attn * attn_loss + diff_loss
                loss = self.lambda_attn * attn_loss + self.lambda_diff * diff_loss
                # step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # finish a batch
                self.diff_recorder.update(diff_loss.item(), cur_batch)
                self.attn_recorder.update(attn_loss.item(), cur_batch)
            # finish all batch/finish a epoch
            lr_scheduler.step()
        # finish training
        lora_state_dict = {k:v for k, v in self.peft_model.unet.state_dict().items() if 'lora_' in k.lower()}
        torch.save(lora_state_dict, self.lora_path)
        self.peft_model.unet.eval()
                    