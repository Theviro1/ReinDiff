import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Literal
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from typing import Literal
from model.stable_diffusion import StableDiffusion
from nltk.corpus import wordnet


scaling_factor = 0.18215
guidance_scale = 7.5
transformer = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
device = 'cuda'
dtype = torch.float32


def calc_t(scheduler:PNDMScheduler, t:int):
    timesteps = scheduler.timesteps
    return timesteps[-t]

def get_synonyms(word:str):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def read_image(path:str, type:Literal['Tensor', 'Image', 'Numpy'], convert:bool=False, unsqueeze:bool=False)->torch.Tensor|Image.Image|np.ndarray:
    image = Image.open(path).resize((512, 512), resample=Image.BILINEAR)
    # convert
    if convert:
        image = image.convert('RGB')
    # type
    if type == 'Tensor':
        # if convert an extra dimension would be 3, unsqueeze returns (1, 3, 512, 512), else returns (3, 512, 512)
        if unsqueeze:
            image = transformer(image).unsqueeze(0).to(device=device, dtype=dtype)
        else:
            image = transformer(image).to(device=device, dtype=dtype)
    elif type == 'Numpy':
        image = np.array(image)
    return image

def reformat_image(images:torch.Tensor|np.ndarray):
    # convert images from torch.Tensor to PIL.Image, images shaped as (batch_size, 512, 512, 3)
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if images.dtype != np.uint8:
        images = (images * 255).clip(0, 255).astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def cnt_tokens(tokenizer:CLIPTokenizer, prompt:str):
    text_inputs = tokenizer(prompt, padding=False, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    return text_input_ids.shape[1]

def get_tokens(tokenizer:CLIPTokenizer, prompt:str):
    text_inputs = tokenizer(prompt, padding=False, truncation=True, return_tensors=None)
    text_inputs_ids = text_inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(text_inputs_ids)
    tokens = [str(token).rstrip('</w>') for token in tokens]
    return tokens


def encode_prompt(pipe:StableDiffusionPipeline, prompt:str|list[str]):
    prompt_embed, empty_prompt_embed = pipe.encode_prompt(prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, device=device)
    return torch.cat([empty_prompt_embed, prompt_embed]).to(device=device, dtype=dtype)

def encode_image(vae:AutoencoderKL, image:torch.Tensor):
    # vae encoded value on every pixel is a little high, thus multiply a scaling_factor to fit unet input
    latent = vae.encode(image).latent_dist.sample() * scaling_factor
    return latent

def latent_t_rev(model:StableDiffusion, embeddings:torch.Tensor, t_ddim:int)->torch.Tensor:
    rand_noise = torch.randn((int(embeddings.shape[0]/2), 4, 64, 64), dtype=dtype, device=device)
    latents = rand_noise
    model.scheduler.counter = 0
    model.scheduler.ets = []
    for i, t in enumerate(model.scheduler.timesteps[:-t_ddim]):
        noise_preds = predict(model.unet, latents, t, embeddings)
        latent_preds = step(model.scheduler, latents, t, noise_preds)
        latents = latent_preds
    return latents   # (batch_size, 4, 64, 64) at timesteps t_ddim


def latent_t_comb(model:StableDiffusion, embeddings:torch.Tensor, t_ddim:int, num_steps:int=9)->torch.Tensor:
    rand_noise = torch.randn((int(embeddings.shape[0]/2), 4, 64, 64), dtype=dtype, device=device)
    latents = rand_noise
    indices = np.linspace(0, t_ddim, num_steps, dtype=int).tolist()
    for i in range(len(indices)-1):
        t = calc_t(model.scheduler, indices[i])
        noise_preds = predict(model.unet, latents, t, embeddings)
        t_prev = calc_t(model.scheduler, indices[i+1])
        alpha_t = model.scheduler.alphas_cumprod[-t]
        alpha_t_prev = model.scheduler.alphas_cumprod[-t_prev]
        x0 = (latents - (1 - alpha_t).sqrt() * noise_preds) / alpha_t.sqrt()
        latents = alpha_t_prev.sqrt() * x0 + (1 - alpha_t_prev).sqrt() * noise_preds
    return latents


def latent_t_dir(model:StableDiffusion, embeddings:torch.Tensor, t_ddim:int)->torch.Tensor:
    rand_noise = torch.randn((int(embeddings.shape[0]/2), 4, 64, 64), dtype=dtype, device=device)
    latents = rand_noise
    t = model.scheduler.timesteps[0]
    t_tar = calc_t(model.scheduler, t_ddim)
    noise_preds = predict(model.unet, latents, t, embeddings)
    alpha_t = model.scheduler.alphas_cumprod[-t]
    alpha_t_tar = model.scheduler.alphas_cumprod[-t_tar]
    x0 = (latents - (1 - alpha_t).sqrt() * noise_preds) / alpha_t.sqrt()
    latents_tar = alpha_t_tar.sqrt() * x0 + (1 - alpha_t_tar).sqrt() * noise_preds
    return latents_tar



def decode_latent(vae:AutoencoderKL, latent):
    # latent support batch_size shaped as (batch_size, c, h, w)
    image = vae.decode(latent / scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).to(dtype=dtype)
    return image

def add_noise(scheduler:PNDMScheduler, latent, t, epsilon=None):
    if epsilon is None:
        epsilon = torch.randn_like(latent, dtype=dtype).to(device)
    alpha = scheduler.alphas_cumprod[t]
    noise_latent = (alpha**0.5) * latent + ((1 - alpha)**0.5) * epsilon
    return epsilon, noise_latent

def predict(unet:UNet2DConditionModel, latent, t, embedding):
    latent_cfg = torch.cat([latent] * 2)
    noise_pred = unet(latent_cfg, t, encoder_hidden_states=embedding).sample.to(device)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def step(scheduler:PNDMScheduler, latent, t, noise_pred):
    latent_pred = scheduler.step(noise_pred, t, latent).prev_sample.to(device)
    return latent_pred


def step_logprob(scheduler:PNDMScheduler, latent, t, noise_pred:torch.Tensor, t_prev, eta)->tuple[torch.Tensor, torch.Tensor]:
    dtype, device = noise_pred.dtype, noise_pred.device
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_prev_t = scheduler.alphas_cumprod[t_prev]
    beta_prod_t = 1 - alpha_prod_t
    std_dev_t = (eta * torch.sqrt((1-alpha_prod_prev_t) / (1-alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_prev_t))).to(dtype=dtype, device=device)
    x0 = (latent - torch.sqrt(beta_prod_t)*noise_pred) / torch.sqrt(alpha_prod_t)
    prev_sample = torch.sqrt(alpha_prod_prev_t) * x0 + torch.sqrt(1 - alpha_prod_prev_t - std_dev_t**2) * noise_pred

    z = torch.randn_like(latent)
    next_latent = prev_sample + std_dev_t * z
    dist = torch.distributions.Normal(prev_sample, std_dev_t)
    log_prob = dist.log_prob(next_latent).mean(dim=[1, 2, 3])
    return  next_latent, log_prob


def origin(scheduler:PNDMScheduler, latent, t, noise_pred):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    x0_latent = (latent - (beta_prod_t ** 0.5)*noise_pred) / (alpha_prod_t ** 0.5)
    return x0_latent

def resample(image:torch.Tensor, size:tuple=(64, 64)) -> torch.Tensor:
    # image shape like (n, n)
    h, w = image.shape
    if (h, w) == size:
        return image
    image_reshaped = image.unsqueeze(0).unsqueeze(0)
    image_downsampled = torch.nn.functional.interpolate(image_reshaped, size=size, mode='bilinear', align_corners=False)
    image = image_downsampled.squeeze(0).squeeze(0)
    return image






