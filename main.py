from config.training_config import *
from config.default_config import *
from lora.peft_model import get_peft_model, load_lora_weight, enable_lora, disable_lora
from model.mask2former import Mask2Former
from model.stable_diffusion import StableDiffusion
from sdlib import dataset_utils, diffusion_utils, visualize_utils
from lora.lora_sft import DiffusionLoraSFTTrainer
from lora.lora_rl_regular_reward import DiffusionLoraRLRegularTrainer
from lora.lora_rl_m2f_reward import DiffusionLoraRLM2FTrainer
import torch
import os


def gen(
    random_noise:torch.Tensor,
    prompt:str,
    save_folder:str,
):
    name = 'original'
    print(f'original model generating image...')
    stable_diffusion = StableDiffusion()
    t_ddim = 50
    tokens = diffusion_utils.get_tokens(stable_diffusion.tokenizer, prompt)
    embeddings = diffusion_utils.encode_prompt(stable_diffusion.pipe, prompt)
    noise_latent = random_noise
    stable_diffusion.scheduler.ets = []
    stable_diffusion.scheduler.counter = 0
    for i, t in enumerate(stable_diffusion.scheduler.timesteps[-t_ddim:]):
        noise_pred = diffusion_utils.predict(stable_diffusion.unet, noise_latent, t, embeddings)
        latent_pred = diffusion_utils.step(stable_diffusion.scheduler, noise_latent, t, noise_pred)
        noise_latent = latent_pred
    img = diffusion_utils.decode_latent(stable_diffusion.vae, noise_latent)[0]
    attn_scores = stable_diffusion.attn_scores
    path_lora_img = os.path.join(save_folder, name+'.png')
    path_lora_attnmap = os.path.join(save_folder, name + '_attnmap.png')
    visualize_utils.visualize_image(img, path_lora_img)
    visualize_utils.visualize_merged_attn_score(attn_scores, tokens, save_path=path_lora_attnmap)

def rev(
    img_path:str, 
    random_noise:torch.Tensor,
    prompt:str, 
    t_ddim:int, 
    save_folder:str, 
 ):
    name = 'original'
    print(f'original model reversing image...')
    img = diffusion_utils.read_image(img_path, 'Tensor', unsqueeze=True)  # convert to batch
    stable_diffusion = StableDiffusion()
    latent = diffusion_utils.encode_image(stable_diffusion.vae, img)
    _, noise_latent = diffusion_utils.add_noise(stable_diffusion.scheduler, latent, diffusion_utils.calc_t(stable_diffusion.scheduler, t_ddim), random_noise)
    tokens = diffusion_utils.get_tokens(stable_diffusion.tokenizer, prompt)
    embeddings = diffusion_utils.encode_prompt(stable_diffusion.pipe, prompt)
    stable_diffusion.scheduler.ets = []
    stable_diffusion.scheduler.counter = 0
    for i, t in enumerate(stable_diffusion.scheduler.timesteps[-t_ddim:]):
        noise_pred = diffusion_utils.predict(stable_diffusion.unet, noise_latent, t, embeddings)
        latent_pred = diffusion_utils.step(stable_diffusion.scheduler, noise_latent, t, noise_pred)
        noise_latent = latent_pred
    img = diffusion_utils.decode_latent(stable_diffusion.vae, noise_latent)[0]
    attn_scores = stable_diffusion.attn_scores
    path_lora_img = os.path.join(save_folder, name+'.png')
    path_lora_attnmap = os.path.join(save_folder, name + '_attnmap.png')
    visualize_utils.visualize_image(img, path_lora_img)
    visualize_utils.visualize_merged_attn_score(attn_scores, tokens, save_path=path_lora_attnmap)



def lora_gen(
    random_noise:torch.Tensor, 
    prompt:str, 
    lora_range:tuple[int, int], 
    lora_weight_path:str, 
    save_folder:str, 
    name:str
):
    print(f'lora {name} generating image...')
    stable_diffusion = StableDiffusion()
    get_peft_model(stable_diffusion)
    t_ddim = 50
    tokens = diffusion_utils.get_tokens(stable_diffusion.tokenizer, prompt)
    embeddings = diffusion_utils.encode_prompt(stable_diffusion.pipe, prompt)
    noise_latent = random_noise
    hook_lora_step = t_ddim - lora_range[1]
    remove_lora_step = t_ddim - lora_range[0]
    stable_diffusion.scheduler.ets = []
    stable_diffusion.scheduler.counter = 0
    for i, t in enumerate(stable_diffusion.scheduler.timesteps[-t_ddim:]):
        if i == hook_lora_step-1:
            load_lora_weight(stable_diffusion, lora_weight_path)
        if i == remove_lora_step-1:
            disable_lora(stable_diffusion)
        noise_pred = diffusion_utils.predict(stable_diffusion.unet, noise_latent, t, embeddings)
        latent_pred = diffusion_utils.step(stable_diffusion.scheduler, noise_latent, t, noise_pred)
        noise_latent = latent_pred
    img = diffusion_utils.decode_latent(stable_diffusion.vae, noise_latent)[0]
    attn_scores = stable_diffusion.attn_scores
    path_lora_img = os.path.join(save_folder, name+'.png')
    path_lora_attnmap = os.path.join(save_folder, name + '_attnmap.png')
    visualize_utils.visualize_image(img, path_lora_img)
    visualize_utils.visualize_merged_attn_score(attn_scores, tokens, save_path=path_lora_attnmap)

def lora_rev(
    img_path:str, 
    random_noise:torch.Tensor, 
    prompt:str,
    t_ddim:int, 
    lora_range:tuple[int, int], 
    lora_weight_path:str, 
    save_folder:str, 
    name:str
):
    print(f'lora {name} reversing image...')
    img = diffusion_utils.read_image(img_path, 'Tensor', unsqueeze=True) # convert to batch
    stable_diffusion = StableDiffusion()
    get_peft_model(stable_diffusion)
    latent = diffusion_utils.encode_image(stable_diffusion.vae, img)
    _, noise_latent = diffusion_utils.add_noise(stable_diffusion.scheduler, latent, diffusion_utils.calc_t(stable_diffusion.scheduler, t_ddim), epsilon=random_noise)
    hook_lora_step = t_ddim - lora_range[1]
    remove_lora_step = t_ddim - lora_range[0]
    tokens = diffusion_utils.get_tokens(stable_diffusion.tokenizer, prompt)
    embeddings = diffusion_utils.encode_prompt(stable_diffusion.pipe, prompt)
    stable_diffusion.scheduler.ets = []
    stable_diffusion.scheduler.counter = 0
    if hook_lora_step <= 0:
        load_lora_weight(stable_diffusion, lora_weight_path)
    for i, t in enumerate(stable_diffusion.scheduler.timesteps[-t_ddim:]):
        if i == hook_lora_step-1:
            load_lora_weight(stable_diffusion, lora_weight_path)
        if i == remove_lora_step-1:
            disable_lora(stable_diffusion)
        noise_pred = diffusion_utils.predict(stable_diffusion.unet, noise_latent, t, embeddings)
        latent_pred = diffusion_utils.step(stable_diffusion.scheduler, noise_latent, t, noise_pred)
        noise_latent = latent_pred
    img = diffusion_utils.decode_latent(stable_diffusion.vae, noise_latent)[0]
    attn_scores = stable_diffusion.attn_scores
    path_lora_img = os.path.join(save_folder, name+'.png')
    path_lora_attnmap = os.path.join(save_folder, name + '_attnmap.png')
    visualize_utils.visualize_image(img, path_lora_img)
    visualize_utils.visualize_merged_attn_score(attn_scores, tokens, save_path=path_lora_attnmap)









def test_gen(prompt:str):
    save_folder = OUTPUT_FOLDER_PATH
    lora_range = LORA_RANGE
    with torch.no_grad():
        random_noise = torch.randn((1, 4, 64, 64), dtype=torch.float32, device='cuda')
        random_noise_path = os.path.join(save_folder, 'random_noise.pt')
        torch.save(random_noise, random_noise_path)
        gen(random_noise, prompt, save_folder)
        lora_gen(random_noise, prompt, lora_range, LORA_SFT_PATH, save_folder, 'sft')
        lora_gen(random_noise, prompt, lora_range, LORA_RL_REGULAR_PATH, save_folder, 'rl_regular')
        lora_gen(random_noise, prompt, lora_range, LORA_RL_M2F_PATH, save_folder, 'rl_m2f')


def test_rev(prompt:str, t_ddim:int):
    img_path = DEMO_PATH
    lora_range = LORA_RANGE
    save_folder = REVERSE_FOLDER_PATH
    with torch.no_grad():
        random_noise = torch.randn((1, 4, 64, 64), dtype=torch.float32, device='cuda')
        random_noise_path = os.path.join(save_folder, 'random_noise.pt')
        torch.save(random_noise, random_noise_path)
        rev(img_path, random_noise, prompt, t_ddim, save_folder)
        lora_rev(img_path, random_noise, prompt, t_ddim, lora_range, LORA_SFT_PATH, save_folder, 'sft')
        lora_rev(img_path, random_noise, prompt, t_ddim, lora_range, LORA_RL_REGULAR_PATH, save_folder, 'rl_regular')
        lora_rev(img_path, random_noise, prompt, t_ddim, lora_range, LORA_RL_M2F_PATH, save_folder, 'rl_m2f')


def train_sft():
    stable_diffusion = StableDiffusion()
    config = SFT_CONFIG
    stable_diffusion = get_peft_model(stable_diffusion)
    trainer = DiffusionLoraSFTTrainer(stable_diffusion, **config)
    trainer.train()

def train_rl_regular(from_sft:bool=False):
    stable_diffusion = StableDiffusion()
    config = RL_REGULAR_CONFIG
    stable_diffusion = get_peft_model(stable_diffusion)
    if from_sft:
        load_lora_weight(stable_diffusion, LORA_SFT_PATH)
    trainer = DiffusionLoraRLRegularTrainer(stable_diffusion, **config)
    trainer.train()

def train_rl_regular_chain(from_sft:bool=False):
    stable_diffusion = StableDiffusion()
    config = RL_REGULAR_CHAIN_CONFIG
    stable_diffusion = get_peft_model(stable_diffusion)
    if from_sft:
        load_lora_weight(stable_diffusion, LORA_SFT_PATH)
    trainer = DiffusionLoraRLRegularTrainer(stable_diffusion, **config)
    trainer.unroll_chain_train()

def train_rl_m2f(from_sft:bool=False):
    stable_diffusion = StableDiffusion()
    mask2former = Mask2Former()
    config = RL_M2F_CONFIG
    stable_diffusion = get_peft_model(stable_diffusion)
    if from_sft:
        load_lora_weight(stable_diffusion, LORA_SFT_PATH)
    trainer = DiffusionLoraRLM2FTrainer(stable_diffusion, mask2former, **config)
    trainer.train()

def train_rl_m2f_chain(from_sft:bool=False):
    stable_diffusion = StableDiffusion()
    config = RL_M2F_CHAIN_CONFIG
    stable_diffusion = get_peft_model(stable_diffusion)
    if from_sft:
        load_lora_weight(stable_diffusion, LORA_SFT_PATH)
    trainer = DiffusionLoraRLM2FTrainer(stable_diffusion, **config)
    trainer.unroll_chain_train()

def download_dataset(start_idx, end_idx):
    dataset_utils.download(start_idx, end_idx)






# -----------------------------
prompt = 'cat, traffic light, baseball bat, dining table'
# prompt = 'A bookshelf stacked with books, a golden cat laying in sofa, a clock on the wall, a potted plant in the corner, a cosmic landscape photo hanging on the wall'
prompt = 'A person eating meals with fork on dining table, a potted plant in the corner.'
prompt = 'A cat, a dog and a man'

# train_sft()
test_gen(prompt)
# train_rl_regular(from_sft=True)
# train_rl_m2f(from_sft=True)
