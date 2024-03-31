from typing import Dict
from Diffusion.Model import UNetModel
from Diffusion import gaussian_diffusion as gd
from Diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def make_beta_schedule(schedule, n_timestep, betas_1=1e-4, beats_T=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = torch.linspace(betas_1 ** 0.5, beats_T ** 0.5,
                            n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'linear':
        betas = torch.linspace(betas_1, beats_T,
                            n_timestep, dtype=torch.float64)
    elif schedule == 'const':
        betas = beats_T * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(n_timestep,
                                 1, n_timestep, dtype=torch.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def create_model_and_diffusion(modelConfig: Dict):
    model = create_model(
        in_channels=modelConfig["in_channels"],
        out_channels=modelConfig["out_channels"],
        image_size=modelConfig["image_size"],
        num_channels=modelConfig["num_channels"],
        num_res_blocks=modelConfig["num_res_blocks"],
        channel_mult=modelConfig["channel_mult"],
        learn_sigma=modelConfig["learn_sigma"],
        use_checkpoint=modelConfig["use_checkpoint"],
        attention_resolutions=modelConfig["attention_resolutions"],
        num_heads=modelConfig["num_heads"],
        num_head_channels=modelConfig["num_head_channels"],
        num_heads_upsample=modelConfig["num_heads_upsample"],
        use_scale_shift_norm=modelConfig["use_scale_shift_norm"],
        dropout=modelConfig["dropout"],
        resblock_updown=modelConfig["resblock_updown"],
        use_fp16=modelConfig["use_fp16"],
        use_new_attention_order=modelConfig["use_new_attention_order"],
    )
    diffusion = create_gaussian_diffusion(
        sample=modelConfig["sample_method"],
        steps=modelConfig["diffusion_steps"],
        beta_1=modelConfig["beta_1"],
        beta_T=modelConfig["beta_T"],
        learn_sigma=modelConfig["learn_sigma"],
        noise_schedule=modelConfig["noise_schedule"],
        use_kl=modelConfig["use_kl"],
        predict_xstart=modelConfig["predict_xstart"],
        rescale_timesteps=modelConfig["rescale_timesteps"],
        rescale_learned_sigmas=modelConfig["rescale_learned_sigmas"],
        timestep_respacing=modelConfig["timestep_respacing"],
    )
    return model, diffusion


def create_Unet_model(modelConfig: Dict):
    model = create_model(
        in_channels=modelConfig["in_channels"],
        out_channels=modelConfig["out_channels"],
        image_size=modelConfig["image_size"],
        num_channels=modelConfig["num_channels"],
        num_res_blocks=modelConfig["num_res_blocks"],
        channel_mult=modelConfig["channel_mult"],
        learn_sigma=modelConfig["learn_sigma"],
        use_checkpoint=modelConfig["use_checkpoint"],
        attention_resolutions=modelConfig["attention_resolutions"],
        num_heads=modelConfig["num_heads"],
        num_head_channels=modelConfig["num_head_channels"],
        num_heads_upsample=modelConfig["num_heads_upsample"],
        use_scale_shift_norm=modelConfig["use_scale_shift_norm"],
        dropout=modelConfig["dropout"],
        resblock_updown=modelConfig["resblock_updown"],
        use_fp16=modelConfig["use_fp16"],
        use_new_attention_order=modelConfig["use_new_attention_order"],
    )
    return model

def create_model(
    image_size,
    in_channels,
    num_channels,
    out_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    text_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(out_channels if not learn_sigma else out_channels*2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

NUM_CLASSES = 1000
def create_gaussian_diffusion(
    *,
    sample=None,
    steps=1000,
    beta_1 = 1e-4,
    beta_T = 0.02,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = make_beta_schedule(schedule=noise_schedule, n_timestep=steps, betas_1=beta_1, beats_T=beta_T)

    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if sample=='DDPM':
        timestep_respacing=None,
    else:
        timestep_respacing=timestep_respacing
        if not timestep_respacing:
            timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def disp(img):
    if img.shape[0] == 1:
        img = img.cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img =np.transpose(img, (1, 2, 0))
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # 可以去除坐标轴
        plt.show()
    else:
        img = img.cpu().numpy()
        # img = (img * 255).astype(np.uint8)
        selected_channels = [0, 1, 2]
        img = img[selected_channels, :, :]
        img =np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')  # 可以去除坐标轴
        plt.show()
