import math
import time
import torch
import random
from Diffusion.util import create_model_and_diffusion
from Diffusion.misc import set_random_seed
from torchvision import utils
from typing import Dict
from libtiff import TIFF
import numpy as np
from torch.nn import functional as F
from Diffusion.resizer import Resizer

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_cond_img(ms_path, pan_path):

    ms4_tif = TIFF.open(ms_path, mode='r')
    ms4_np = ms4_tif.read_image()
    # print(ms4_np)
    print('原始ms4图的形状：', np.shape(ms4_np))

    pan_tif = TIFF.open(pan_path, mode='r')
    pan_np = pan_tif.read_image()
    # print(pan_np)
    print('原始pan图的形状;', np.shape(pan_np))
    """crop image to 128x128"""

    # ms4_np = ms4_np[3125:3253, 3125:3253, :]
    # pan_np = pan_np[12500:13012, 12500:13012]

    ms4_np = ms4_np[0:128, 0:128, :]
    pan_np = pan_np[0:512, 0:512]

    '''归一化图片'''
    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)

    pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
    ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道
    # print(ms4.shape)
    # disp(ms4)

    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)


    ms4 = ms4.unsqueeze(0)  # 增加一个维度
    ms4 = F.interpolate(ms4, scale_factor=4, mode='bicubic', align_corners=False)
    pan = pan.unsqueeze(0)  # 增加一个维度

    print(ms4.shape)
    print(pan.shape)
    return ms4, pan



def sample(modelConfig: Dict):
    time0 = time.time()

    if modelConfig["randomized_seed"]:
        modelConfig["seed"] = random.randint(0, 10000)
    set_random_seed(modelConfig["seed"], by_rank=True)

    tb_log = None

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

    shape = (1, 4, 512, 512)
    shape_d = (1, 4, int(512 / modelConfig["down_N"]),
               int(512 / modelConfig["down_N"]))
    down = Resizer(shape, 1 / modelConfig["down_N"]).to(next(Unet_model.parameters()).device)
    up = Resizer(shape_d, modelConfig["down_N"]).to(next(Unet_model.parameters()).device)
    resizers = (down, up)


    # create data
    pan_path = modelConfig["pan_path"]
    ms_path = modelConfig["ms_path"]

    ms, pan = load_cond_img(ms_path, pan_path)

    ms = ms.to('cuda')
    pan = pan.to('cuda')


    import matplotlib.pyplot as plt

    def disp(img):

        img = img.squeeze(0)  # 删除一个维度

        if img.shape[0] == 1:
            img = img.cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img, cmap='gray')
            plt.axis('off')  # 可以去除坐标轴
            plt.show()
        else:
            img = img.cpu().numpy()
            img = (img * 255).astype(np.uint8)
            selected_channels = [0, 1, 2]
            img = img[selected_channels, :, :]
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            plt.axis('off')  # 可以去除坐标轴
            plt.show()

    disp(pan)
    disp(ms)
    disp(ms+pan)

    print("creating samples...")
    count = 0
    while count * modelConfig["sample_batch_size"] < modelConfig["num_samples"]:

        sample = diffusion.ddim_sample_loop(
            Unet_model,
            (1, 4, modelConfig["image_size"], modelConfig["image_size"]),
            clip_denoised=modelConfig["clip_denoised"],
            model_kwargs=None,
            resizers=resizers,
            cond_pan=pan,
            cond_ms=ms,
            range_t=modelConfig["range_t"]
        )

        disp(sample)

        for i in range(1):
            out_path = os.path.join("./SampleImgs/",
                                    f"{str(i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )


        count += 1
        # logger.log(f"created {count * args.batch_size} samples")

    # dist.barrier()
    # logger.log("sampling complete")
    print("sampling complete")

