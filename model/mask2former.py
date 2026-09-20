from config.default_config import MASK2FORMER_PATH
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import torch
from PIL import Image

class Mask2Former:
    def __init__(self, model_path:str=MASK2FORMER_PATH):
        self.device = 'cuda'
        self.dtype = torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).eval().to(device=self.device, dtype=self.dtype)

        # reflecting from mask2former output class label to coco2017 official cat_ids (80->90)
        self.cls2gt = [
            1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

    def remap_label(self, instance_results:list[dict])->list[torch.Tensor]:
        instance_remaps = []
        for result in instance_results:
            seg_img = result['segmentation']
            seg_info = result['segments_info']
            min_val = -1
            mapping_arr = np.zeros(len(seg_info)+1, dtype=int)
            mapping_arr[0] = 0  # convert background from -1 to 0
            for label in seg_info:
                mapping_arr[label['id']+1] = self.cls2gt[int(label['label_id'])]  # construct a map in which i(from 0-n) will be replace by j(actual cat_id in coco2017)
            indices = (seg_img.detach().cpu().numpy()-min_val).astype(np.int64)  # each pixel add 1, values from 0-n
            seg_img = torch.from_numpy(np.take(mapping_arr, indices)).to(device=self.device, dtype=self.dtype)  # use map to reconstruct seg_img
            instance_remaps.append(seg_img)
        return torch.stack(instance_remaps).to(dtype=self.dtype, device=self.device)

    def __call__(self, images:torch.Tensor|np.ndarray)->list[torch.Tensor]:
        # image shaped as (batch_size, 512, 512)
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        images = [Image.fromarray((image*255).clip(0, 255).astype(np.uint8)) for image in images]  # batch_size length list of (3, 512, 512) image
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1] for image in images])
            result = self.remap_label(result)  # shaped as (batch_size, 512, 512) as segmentation result
        return result  #returns a tensor shaped as (batch_size, 512, 512)
        