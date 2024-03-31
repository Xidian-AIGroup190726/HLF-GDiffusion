
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm

from Diffusion.train_ddpm_util import GaussianDiffusionTrainer
from Scheduler import GradualWarmupScheduler
from Dataset.dataset import create_train_data_loader
from Diffusion.util import create_Unet_model

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset

    pan_path = modelConfig["pan_path"]
    ms_path = modelConfig["ms_path"]
    fusion_path = modelConfig["fusion_img_path"]
    label_path = modelConfig["label_path"]
    Ms4_patch_size = modelConfig["Ms4_patch_size"]
    BATCH_SIZE = modelConfig["BATCH_SIZE"]
    Train_Rate = modelConfig["Train_Rate"]

    train_loader, test_loader, all_data_loader = create_train_data_loader(pan_path, ms_path, fusion_path, label_path,
                                                                          Ms4_patch_size=Ms4_patch_size,
                                                                          BATCH_SIZE=BATCH_SIZE, Train_Rate=Train_Rate)

    # model setup
    Unet_model = create_Unet_model(modelConfig)

    if modelConfig["training_load_weight"] is not None:
        Unet_model.load_state_dict(
            torch.load(modelConfig["diffusion_model_path"], map_location="cpu")
        )

    optimizer = torch.optim.AdamW(
        Unet_model.parameters(), lr=modelConfig["diffusion_lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["diffusion_epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["diffusion_epoch"] // 10, after_scheduler=cosineScheduler)

    trainer = GaussianDiffusionTrainer(
        model=Unet_model, schedule=modelConfig["noise_schedule"], beta_1=modelConfig["beta_1"], beta_T=modelConfig["beta_T"],
        T=modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["diffusion_epoch"]):
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for ms, pan, _, _, _ in tqdmDataLoader:
                # train_loader 返回 return image_ms, image_pan, target, locate_xy
                optimizer.zero_grad()
                ms_x_0 = ms.to(device)
                pan_x_0 = pan.to(device)

                pan_x_0 = pan_x_0.repeat(1, 4, 1, 1)
                loss = trainer(pan_x_0).sum() / 1000.  # 除以T
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    Unet_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()

        torch.save(Unet_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

        # if e % 10 == 0:
        #     torch.save(Unet_model.state_dict(), os.path.join(
        #         modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

