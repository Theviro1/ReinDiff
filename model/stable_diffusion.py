from diffusers import StableDiffusionPipeline
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
import math
from diffusers.models.attention_processor import Attention, AttnProcessor
from typing import Optional
from config.default_config import STABLE_DIFFUSION_PATH
from config.training_config import LAYER_WEIGHT


class CustomAttnProcessor(AttnProcessor):
    def __init__(self):
        super().__init__()
        self.attn_score = None
    
    def calc_attn_logits(
            self, query:torch.Tensor, key:torch.Tensor, attention_mask:torch.Tensor = None, head_dim:int=64
        ) -> torch.Tensor:
        # query and key doesn't split to many heads currently
        dtype = query.dtype
        dim_head = head_dim
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
        scale = dim_head**-0.5
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=scale,
        )
        del baddbmm_input
        attention_scores = attention_scores.to(dtype)
        return attention_scores
    
    def calc_attn_scores(
            self, query:torch.Tensor, key:torch.Tensor, attention_mask:torch.Tensor = None,
        ) -> torch.Tensor:
        # query shaped as (2*batch_size, h*w, d_attn)
        # key shaped as (2*batch_size, n, d_attn)
        scale = 1.0 / math.sqrt(query.shape[-1])
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        return attention_scores  # attention_scores shaped as (2*batch_size, h*w, n)

    def __call__(
            self,
            attn:Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args, 
            **kwargs
        )->torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            # batch input hidden_states shaped as (2*batch_size, c, h, w), reshape to (2*batch_size, h*w, c)
            # first (0~batch_size-1, h*w, c) is about CFG empty prompt, last (batch_size~2*batch_size-1, h*w, c) is real prompt
            # same image should visit hidden_states[i] and hidden_states[i+batch_size] as CFG and real
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        # (2*batch_size, h*w, c) @ (c, d_attn) -> (2*batch_size, h*w, d_attn)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        # self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        # encoder_hidden_states also (2*batch_size, n, d)
        # (2*batch_size, n, d) @ (d, d_attn) -> (2*batch_size, n, d_attn)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        '''
        hidden state有两个维度分别是Classifier free的和真实的,因此to_q to_k to_v之后也是两个维度,只有第2个维度是有用的,添加一行代码：
        此外上面的to_q to_k to_v都是已经注入了lora的结果,因此计算图中可以囊括lora模块并且拿到的也是lora之后的attention score
        '''
        # self.attn_score = self.calc_attn_logits(query, key, attention_mask, head_dim)[1]
        real_batch_size = int(batch_size / 2)
        self.attn_score = self.calc_attn_scores(query=query, key=key, attention_mask=None)[real_batch_size:]  # shaped as (batch_size, h*w, n)


        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StableDiffusion:
    def __init__(
        self, 
        model_path:str=STABLE_DIFFUSION_PATH, 
        num_inference_steps:int=50,
        layer_weight:list[float] = LAYER_WEIGHT
    ):
        self.device = 'cuda'
        # self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variant='fp16').to('cuda')
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to('cuda')
        self.layer_weight = layer_weight

        # Downblocks
        for i, downblock in enumerate(self.pipe.unet.down_blocks):
            if hasattr(downblock, 'attentions'):
                for attn_module in downblock.attentions:
                    for trfms_module in attn_module.transformer_blocks:
                        # list of BasicTransformerBlock
                        trfms_module.attn2.set_processor(CustomAttnProcessor()) # cross-attn
        # Midblocks
        if hasattr(self.pipe.unet.mid_block, 'attentions'):
            for attn_module in self.pipe.unet.mid_block.attentions:
                for trfms_module in attn_module.transformer_blocks:
                    # list of BasicTransformerBlock
                    trfms_module.attn2.set_processor(CustomAttnProcessor())  # cross-attn
        # Upblocks
        for i, upblock in enumerate(self.pipe.unet.up_blocks):
            if hasattr(upblock, 'attentions'):
                for attn_module in upblock.attentions:
                    for trfms_module in attn_module.transformer_blocks:
                        # list of BasicTransformerBlock
                        trfms_module.attn2.set_processor(CustomAttnProcessor())  # cross-attn

        self.pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    @property
    def attn_scores(self)->list[torch.Tensor]:
        # returns a list(different layers) of (batch_size, n, v, v) attention scores
        attn_scores = []
        for name, module in self.pipe.unet.named_modules():
            if isinstance(module, Attention) and isinstance(module.get_processor(), CustomAttnProcessor):
                attn_score = module.get_processor().attn_score
                batch_size, len_pixel, len_token = attn_score.shape  # (batch_size, h*w, n)
                attn_score = attn_score.permute(0, 2, 1).reshape(batch_size, len_token, int(math.sqrt(len_pixel)), int(math.sqrt(len_pixel))) # (batch_size, n, v, v)
                attn_scores.append(attn_score)
        return attn_scores

    @property
    def merged_attn_scores(self)->torch.Tensor:
        # merge all attn_scores in different layers, returns a tensor shaped as (77, 64, 64)
        attn_scores = self.attn_scores  # list of Tensors shaped as (batch_size, 77, h, w)
        batch_size, num_tokens, _, _ = attn_scores[0].shape
        merged_attn_scores = torch.zeros((batch_size, num_tokens, 64, 64), dtype=torch.float32, device=self.device)
        for layer_index, layer_attn_score in enumerate(attn_scores):
            # layer_attn_score shaped as (batch_size, num_tokens, h, w)
            _, _, h, w = layer_attn_score.shape
            if (h, w) != (64, 64):
                layer_attn_score = torch.nn.functional.interpolate(layer_attn_score, size=(64, 64), mode='bilinear', align_corners=False)
            merged_attn_scores += self.layer_weight[layer_index] * layer_attn_score
        return merged_attn_scores  # shaped as (batch_size, 77, 64, 64)
    
    @property
    def unet(self)->UNet2DConditionModel:
        return self.pipe.unet
    
    @property
    def scheduler(self)->PNDMScheduler:
        return self.pipe.scheduler
    
    @property
    def tokenizer(self)->CLIPTokenizer:
        return self.pipe.tokenizer
    
    @property
    def vae(self)->AutoencoderKL:
        return self.pipe.vae

