from Scripts.Train_Diffusion import train
from Scripts.sample import sample
from Scripts.train_cls import train_cls
from Scripts.save_fusion_image import generate_image
from Scripts.color import color

def main(model_config = None):
    modelConfig = {
        # train
        "state": "train_cls", # train_diffusion or sample or train_cls or generate_img or color
        "BATCH_SIZE": 32,
        "diffusion_epoch": 100,
        "cls_epoch": 20,
        "use_fp16": False,
        "fp16_hyperparams": "pytorch",
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "diffusion_lr": 1e-4,
        "cls_lr": 1e-3,
        "anneal_lr": False,
        "lr_anneal_steps": 0,
        "training_load_weight": None,
        "multiplier": 2.,
        "T": 1000,
        "iterations": 500000,
        "schedule_sampler": "uniform",

        # sample
        "sample_method": "DDIM", # DDPM or DDIM
        "randomized_seed": True,
        "sample_batch_size": 4,
        "num_samples": 1,
        "seed": 2,
        "single_gpu": False,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        # "device": "cpu",
        "image_weight": 100,
        "clip_denoised": True,

        # Unet
        "class_cond": False,
        "text_cond": False,
        "in_channels": 4,
        "num_channels": 128,
        "out_channels": 4,
        "attention_resolutions": "16,8",
        "conv_resample": True,
        "channel_mult": "",
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "dropout": 0.15,
        "num_res_blocks": 2,
        "use_checkpoint": False,
        "grad_clip": 1.,

        # "attn": [2],
        # "nrow": 8,

        # diffusion
        "learn_sigma": False,
        "diffusion_steps": 2000,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "timestep_respacing": [1000], # 经过respacing后还需采样多少步
        "rescale_learned_sigmas": False,

        # resizers
        "down_N": 2,
        "range_t": 0,


        #data
        "pan_path": "Dataset/data/bigXian/pan.tif",
        "ms_path": "Dataset/data/bigXian/ms4.tif",
        "fusion_img_path": "Dataset/data/bigXian/xian_fusion.tif",
        # "fusion_img_path": None,
        # "label_path": "Dataset/data/Xian/label.npy",
        "train_label_path": "Dataset/data/bigXian/train.npy",
        "test_label_path": "Dataset/data/bigXian/test.npy",
        "Ms4_patch_size": 16,
        "Train_Rate": 0.1, # 训练集划分
        "image_size": 64,

        # save
        "save_weight_dir": "./Checkpoints/",
        "diffusion_model_path": "./Checkpoints/512x512_pan.pt",
        "cls_model_path": "./Checkpoints/bigXian.pt",
        # "cls_model_path": None,
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "logdir": "",
        "exp_name": "guidance_diffusion",
        "log_interval": 10,
        "save_interval": 10000
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train_diffusion":
        train(modelConfig)
    elif modelConfig["state"] == "sample":
        sample(modelConfig)
    elif modelConfig["state"] == "generate_img":
        generate_image(modelConfig)
    elif modelConfig["state"] == "color":
        color(modelConfig)
    else:
        train_cls(modelConfig)


if __name__ == '__main__':
    main()
