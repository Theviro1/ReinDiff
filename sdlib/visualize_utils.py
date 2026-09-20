import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import numpy as np

VISUALIZE_LAYER_WEIGHT = [
    0.01, 0.01,         # 64*64 down-block
    0.01, 0.01,         # 32*32 down-block
    0.01, 0.29,         # 16*16 down-block
    0.29, 0.01, 0.01,   # 16*16 up-block
    0.01, 0.01, 0.01,   # 32*32 up-block
    0.01, 0.01, 0.01,   # 64*64 up-block
    0.29                # 8*8 mid-block
]



def batch_normalize(image:torch.Tensor|np.ndarray):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    img_min = image.min(axis=(1, 2), keepdims=True)
    img_max = image.max(axis=(1, 2), keepdims=True)
    diff = img_max - img_min
    diff[diff < 1e-5] = 1e-5
    return (image - img_min) / diff

def normalize(image:torch.Tensor|np.ndarray):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    img_min = image.min()
    img_max = image.max()
    diff = img_max - img_min
    diff = max(diff, 1e-5)
    return (image - img_min) / diff

def resample(image:torch.Tensor, size:tuple=(64, 64)) -> torch.Tensor:
    # image shape like (n, n)
    h, w = image.shape
    if (h, w) == size:
        return image
    image_reshaped = image.unsqueeze(0).unsqueeze(0)
    image_downsampled = torch.nn.functional.interpolate(image_reshaped, size=size, mode='bilinear', align_corners=False)
    image = image_downsampled.squeeze(0).squeeze(0)
    return image

    
def visualize_attn_score(attn_score:torch.Tensor, tokens:list[str], cmap:str='turbo', save_path:str=None):
    if isinstance(attn_score, torch.Tensor):
        attn_score = attn_score.detach().cpu().numpy()
    # attn_score shaped like (batch_size, 77, n, n), assert batch_size is 1
    assert attn_score.shape[0] == 1, 'batch size over 1, which specific data to be visualized?'
    attn_score = attn_score.astype(np.float32)[0]
    attn_score = batch_normalize(attn_score)

    cols = 5
    num_tokens = len(tokens)
    rows = (num_tokens//cols) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i in range(int(num_tokens)):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        im = attn_score[i]
        im = ax.imshow(im, cmap=cmap)
        fig.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(f'{tokens[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close('all')

@torch.no_grad()
def visualize_merged_attn_score(
    attn_scores:list[torch.Tensor], 
    tokens:list[str],
    layer_weight:list[float] = VISUALIZE_LAYER_WEIGHT,
    cmap:str='turbo', 
    save_path:str=None
):
    num_tokens = len(tokens)
    # attn_score shaped like (batch_size, 77, n, n), assert batch_size is 1
    assert attn_scores[0].shape[0] == 1, 'batch size over 1, which specific data to be visualized?'
    merged_attn_scores = [torch.zeros((64, 64), dtype=torch.float32, device='cuda') for _ in range(num_tokens)]  # supposed to be list of tensor shaped as (77, 64, 64)
    for layer_index, layer_attn_score in enumerate(attn_scores):
        # layer_attn_score shaped as (1, 77, 64, 64)
        layer_attn_score = layer_attn_score[0]
        for i in range(num_tokens):
            token_attn_score = layer_attn_score[i]  # specific token attnmap shaped as (64, 64)
            if token_attn_score.shape != (64, 64):
                token_attn_score = resample(token_attn_score)
            merged_attn_scores[i] += layer_weight[layer_index] * token_attn_score
    merged_attn_scores = torch.stack(merged_attn_scores)

    attn_score = merged_attn_scores.cpu().numpy().astype(np.float32)  # shaped as (77, 64, 64)
    attn_score = batch_normalize(attn_score)

    cols = 5
    rows = (num_tokens//cols) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i in range(num_tokens):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        im = attn_score[i]
        im = ax.imshow(im, cmap=cmap)
        fig.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(f'{tokens[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close('all')

def visualize_image(image, save_path:str=None):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    plt.imshow(image, cmap='turbo')
    plt.axis('off')
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close('all')
    
def visualize_epoch_loss(loss:list[list[float]], save_path:str=None):
    y = [sum(l) for l in loss]
    x = list(range(1, len(y) + 1))

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close('all')
