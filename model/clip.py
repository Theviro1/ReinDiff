from transformers import CLIPModel, CLIPProcessor
from config.default_config import CLIP_PATH
from PIL import Image
import torch
import torch.nn.functional as F

class CLIP:
    def __init__(self, model_path:str=CLIP_PATH):
        self.device = 'cuda'
        self.clip_processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)
        self.clip_model = CLIPModel.from_pretrained(model_path).to(device=self.device)
        self.scaling_factor = 0.1
    
    def cosine_similarities(self, texts:list[str], images:list[Image.Image])->torch.Tensor:
        inputs = self.clip_processor(texts, images, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_embs = self.clip_model.get_image_features(pixel_values=inputs['pixel_values'])
            text_embs = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        image_embs = F.normalize(image_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

        cosine_similarity = image_embs @ text_embs.T
        cosine_similarity = torch.diagonal(cosine_similarity)
        return (1 + cosine_similarity) / 2
    
    def __call__(self, texts:list[str], images:list[Image.Image])->torch.Tensor:
        cos_sim = self.cosine_similarities(texts, images)
        base_sim = self.cosine_similarities([''], images)
        sim = ((cos_sim - base_sim).clamp(0, 1)**self.scaling_factor)
        return sim
        