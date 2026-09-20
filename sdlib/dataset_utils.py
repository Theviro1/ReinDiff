from pycocotools.coco import COCO
from config.default_config import DATASET_PATH, TRAIN_PATH, ANNOTATION_PATH, GT_PATH, CLASS_THRESHOLD
from model.mysql import MySQL
import pickle
from PIL import Image
import os
import io
import skimage
import requests
import numpy as np
import torch
from typing import Any
import traceback

coco = COCO(DATASET_PATH)

categories = coco.loadCats(coco.getCatIds())
sorted_img_ids = sorted(coco.getImgIds())

train_path = TRAIN_PATH
annotation_path = ANNOTATION_PATH
gt_path = GT_PATH

mysql = MySQL()

label2id = {cat['name']: cat['id'] for cat in categories}
id2label = {cat['id']: cat['name'] for cat in categories}


def seg_mask(img_id:int)->np.ndarray:
    # obtain data
    img_info = coco.loadImgs(img_id)[0]
    height, width = img_info['height'], img_info['width']
    ann_id = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_id)
    # define mask
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        raw_seg = ann['segmentation']
        cat_id = ann['category_id']
        for seg in raw_seg:
            poly = np.array(seg).reshape((-1, 2))
            r, c = skimage.draw.polygon(poly[:, 1], poly[:, 0], shape=mask.shape)
            mask[r, c] = cat_id
    return mask


def sep_mask(mask:np.ndarray|torch.Tensor)->list[tuple[int, torch.Tensor]]:
    if isinstance(mask, np.ndarray):
        cat_ids = np.unique(mask)
        cat_ids = cat_ids[cat_ids != 0]
        sep_pairs = []
        for cat_id in cat_ids:
            cat_img = np.where(mask == cat_id, 1, 0)
            sep_pairs.append((int(cat_id), torch.from_numpy(cat_img)))
        return sep_pairs
    elif isinstance(mask, torch.Tensor):
        cat_ids = mask.unique()
        cat_ids = cat_ids[cat_ids != 0]
        sep_pairs = []
        for cat_id in cat_ids:
            cat_img = torch.where(mask == cat_id, 1, 0)
            sep_pairs.append((int(cat_id), cat_img))
        return sep_pairs
    else:
        raise ValueError(f'mask supposed to be torch.Tensor or np.ndarray')


def gen_prompt(sep_pairs:tuple[int, torch.Tensor])->tuple[str, tuple[int, torch.Tensor]]:
    tokens = []
    bias = 3
    prompt_map = []
    for i, (cat_id, cat_img) in enumerate(sep_pairs):
        token = str(id2label[cat_id]).strip()
        tokens.append(token)
        index = bias + i
        prompt_map.append((index, cat_img))
    prompt = ','.join(tokens)
    return prompt, prompt_map

def remap_index(prompt:str, labels:list[tuple[int, Any]])->list[tuple[int, Any]]:
    # target of this function is to eliminate bias caused by mistake in `gen_prompt`
    bias = 3
    startindex = 1
    # labels is a list[(index, cat_img)]
    # prompt like 'x,x x,x'
    tokens = prompt.strip().split(',')  # ['x', 'x x', 'x']
    cur_index = startindex
    target = []
    labels.sort(key = lambda x:x[0])
    for index, cat_img in labels:
        index = index - bias
        # token might consist of a single word or a phrase contains only one space like 'traffic light'
        if ' ' in tokens[index]:
            target.append((cur_index, cat_img))
            cur_index += 1
            target.append((cur_index, cat_img))
        else:
            target.append((cur_index, cat_img))
        cur_index += 1  # commas
        cur_index += 1  # next token index
    return target


def avg_embeddings(prompt:str|list[str], embeddings:torch.Tensor)->torch.Tensor:
    def avg(prompt:str, embeddings:torch.Tensor):
        # prompt should be a single string, embeddings shaped as (77, 768) without CFG
        pos_embed = embeddings  # (77, 768)
        tokens = prompt.strip().split(',')
        cur_index = 1
        for i, token in enumerate(tokens):
            if ' ' in token:
                avg_embed = (pos_embed[cur_index] + pos_embed[cur_index+1]) / 2
                pos_embed[cur_index] = avg_embed
                pos_embed[cur_index+1] = avg_embed
                cur_index += 1  # phrase
            cur_index += 1  # commas
            cur_index += 1  # next token index
    
    # embeddings shaped as (2n, 77, 768) in which only last half is real prompt
    if isinstance(prompt, str):  # n = 1
        avg(prompt, embeddings[1])
        return embeddings
    if isinstance(prompt, list):  # n > 1
        batch_size = len(prompt)
        for b in range(batch_size):
            avg(prompt[b], embeddings[batch_size+b])
        return embeddings

def get_cat_ids(prompt:str, indexs:list[int]):
    # prompt suppose to be 'x, x x, x'
    tokens = prompt.strip().split(',')
    startindex = 3
    desc = [tokens[index - startindex].strip() for index in indexs]
    ids = [coco.getCatIds(catNms=desc)][0]
    assert len(ids) == len(indexs), f'in `{prompt}` using indexs `{indexs}` get `{desc}`, ids `{ids}` which does not match length'
    return ids


def download_image(img_id:int):
    print(f'downloading image {img_id}')
    # preparations
    img_info = coco.loadImgs(img_id)[0]
    url = img_info['coco_url']
    img_path = os.path.join(train_path, str(img_id) + '.png')
    ann_path = os.path.join(annotation_path, str(img_id) + '.pkl')
    try:
        # request for image download
        response = requests.get(url, timeout=5)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        prompt, prompt_map = gen_prompt(sep_mask(seg_mask(img_id)))
        if len(prompt_map) < CLASS_THRESHOLD:
            return
        # save files&infos
        image.save(img_path, format='PNG')
        with open(ann_path, 'wb') as f:
            pickle.dump(prompt_map, f)
        mysql.insert(image_path=img_path, num_classes=len(prompt_map), prompt=prompt, annotation_path=ann_path)
    except:
        traceback.print_exc()


def download(startindex:int=None, endindex:int=None):
    if startindex is None:
        startindex = 0
    if endindex is None:
        endindex = len(sorted_img_ids)
    img_ids = sorted_img_ids[startindex:endindex]
    for img_id in img_ids:
        download_image(img_id)


def add_gt():
    datas = list(mysql.select_all())
    for data in datas:
        id = data['id']
        img_id = int(str(data['image_path']).split('/')[-1].rstrip('.png').strip())
        mask = torch.from_numpy(seg_mask(img_id))
        gt_path_ = os.path.join(gt_path, str(img_id) + '.pt')
        torch.save(mask, gt_path_)  # save
        mysql.insert_column(id, 'gt_path', gt_path_)
    



def select(startindex:int=None, endindex:int=None):
    if startindex is None or endindex is None:
        return list(mysql.select_all())
    else:
        return list(mysql.select_by_index(startindex, endindex))