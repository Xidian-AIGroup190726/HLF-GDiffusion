import time
import torch
from torch.nn import functional as F
import random
from Diffusion.util import create_model_and_diffusion
from Diffusion.misc import set_random_seed
from typing import Dict
from Diffusion.resizer import Resizer
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import math

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


seed = random.randint(0, 10000)
set_random_seed(seed, by_rank=True)


def disp(img):
    if img.shape[0] == 1:
        img = (img * 255).astype(np.uint8)
        img =np.transpose(img, (1, 2, 0))
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # 可以去除坐标轴
        plt.show()
    else:
        img = (img * 255).astype(np.uint8)
        selected_channels = [0, 1, 2]
        img = img[selected_channels, :, :]
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')  # 可以去除坐标轴
        plt.show()

def create_Utils(modelConfig: Dict):

    time0 = time.time()

    if modelConfig["randomized_seed"]:
        modelConfig["seed"] = random.randint(0, 10000)
    set_random_seed(modelConfig["seed"], by_rank=True)

    print("creating model...")

    Unet_model, diffusion = create_model_and_diffusion(modelConfig)

    Unet_model.load_state_dict(
        torch.load(modelConfig["diffusion_model_path"], map_location="cpu")
    )

    Unet_model.to('cuda')

    if modelConfig["use_fp16"]:
        Unet_model.convert_to_fp16()

    Unet_model.eval()

    # create resize
    print("create resizers ...")
    assert math.log(modelConfig["down_N"], 2).is_integer() # 下采样倍数必须是2的整数倍

    shape = (modelConfig["BATCH_SIZE"], 4, modelConfig["image_size"], modelConfig["image_size"])
    shape_d = (modelConfig["BATCH_SIZE"], 4, int(modelConfig["image_size"] / modelConfig["down_N"]),
               int(modelConfig["image_size"] / modelConfig["down_N"]))
    down = Resizer(shape, 1 / modelConfig["down_N"]).to(next(Unet_model.parameters()).device)
    up = Resizer(shape_d, modelConfig["down_N"]).to(next(Unet_model.parameters()).device)
    resizers = (down, up)

    return Unet_model, diffusion, resizers



def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


def generate_image(modelConfig: Dict):

    # 打开原始tif图像
    ms4 = tiff.imread('./Dataset/data/nanjing/ms4.tif')
    pan = tiff.imread('./Dataset/data/nanjing/pan.tif')

    ms4 = to_tensor(ms4)
    pan = to_tensor(pan)

    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)

    width, height, channels = ms4.shape
    width_pan, height_pan = pan.shape

    print("The generate img width and height are ", (width_pan, height_pan))

    block_size = 128  # ms (128, 128)
    block_size_pan = 4 * block_size  # pan (512, 512)

    # (pan_W, pan_H, 4)
    fusion_img = np.zeros((width*4, height*4, channels))

    Unet_model, diffusion, resizers = create_Utils(modelConfig)

    step = math.ceil(width_pan/block_size_pan) * math.ceil(height_pan/block_size_pan)
    print("total need sample: ", step)
    print("stare generate")

    count = 1
    for i in range(0, width, block_size):
        for j in range(0, height, block_size):
            # 获取当前分块的数据，返回一个numpy数组
            if i + block_size > width:
                i = width - block_size
                block_ms = ms4[i:width, j:j+block_size]
                block_pan = pan[i*4:width_pan, j*4:j*4+block_size_pan ]
                block_ms = np.array(block_ms).transpose((2, 0, 1)) # block_ms需要
                block_ms = torch.Tensor(block_ms)

                block_ms = block_ms.unsqueeze(0)  # 增加一个维度
                block_ms = F.interpolate(block_ms, scale_factor=4, mode='bicubic', align_corners=False)

                block_pan = block_pan.unsqueeze(0)  # 增加一个维度
                block_pan = block_pan.unsqueeze(0)


            elif j + block_size > height:
                j = height - block_size
                block_ms = ms4[i:i+block_size, j:height]
                block_pan = pan[i*4:i*4+block_size_pan, j*4:height_pan]
                block_ms = np.array(block_ms).transpose((2, 0, 1)) # block_ms需要
                block_ms = torch.Tensor(block_ms)

                block_ms = block_ms.unsqueeze(0)  # 增加一个维度
                block_ms = F.interpolate(block_ms, scale_factor=4, mode='bicubic', align_corners=False)

                block_pan = block_pan.unsqueeze(0)  # 增加一个维度
                block_pan = block_pan.unsqueeze(0)


            else:
                block_ms = ms4[i:i+block_size, j:j+block_size]
                block_pan = pan[i*4:i*4+block_size_pan, j*4:j*4+block_size_pan]
                block_ms = np.array(block_ms).transpose((2, 0, 1)) # block_ms需要
                block_ms = torch.Tensor(block_ms)

                block_ms = block_ms.unsqueeze(0)  # 增加一个维度
                block_ms = F.interpolate(block_ms, scale_factor=4, mode='bicubic', align_corners=False)

                block_pan = block_pan.unsqueeze(0)  # 增加一个维度
                block_pan = block_pan.unsqueeze(0)

            sample = diffusion.ddim_sample_loop(
                Unet_model,
                (1, 4, 512, 512),
                clip_denoised=modelConfig["clip_denoised"],
                model_kwargs=None,
                resizers=resizers,
                cond_pan=block_pan.to('cuda'),
                cond_ms=block_ms.to('cuda'),
                range_t=modelConfig["range_t"]
            )


            sample = sample.squeeze(0)
            sample = sample.cpu().numpy()
            sample = np.array(sample).transpose((1, 2, 0))  # block_ms需要

            if i + block_size > width:
                i = width - block_size
                fusion_img[i * 4:width_pan, j * 4:j * 4 + block_size_pan] = sample

            elif j + block_size > height:
                j = height - block_size
                fusion_img[i * 4:i * 4 + block_size_pan, j * 4:height_pan] = sample

            else:
                fusion_img[i * 4:i * 4 + block_size_pan, j * 4:j * 4 + block_size_pan] = sample

            print("No." + str(count) + "sample")
            count += 1

    tiff.imwrite('shanghai_fusion.tif', fusion_img)
    print("generate end")
