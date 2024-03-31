from typing import Dict
import numpy as np
import torch
import cv2
from Dataset.dataset import create_color_data_loader
from Diffusion.cls_model import cls_model
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def color(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # create data
    pan_path = modelConfig["pan_path"]
    ms_path = modelConfig["ms_path"]
    fusion_path = modelConfig["fusion_img_path"]
    label_path = modelConfig["label_path"]
    Ms4_patch_size = modelConfig["Ms4_patch_size"]
    BATCH_SIZE = modelConfig["BATCH_SIZE"]
    
    all_data_loader = create_color_data_loader(pan_path, ms_path, fusion_path, label_path, Ms4_patch_size=Ms4_patch_size,
                                                                            BATCH_SIZE=BATCH_SIZE)

    print("create cls model")
    # 此处的output是模型输出，有12类
    model = cls_model(output=12)

    if modelConfig["cls_model_path"] is not None:
        model.load_state_dict(
            torch.load(modelConfig["cls_model_path"], map_location="cpu")
        )

    model.to(device)

    # 上色
    # 和类别多少一样，大西安有12类
    class_count = np.zeros(12)
    # out_color = np.zeros((800, 830, 3))
    # 西安的大小，大小为ms图的大小
    out_color = np.zeros((4541, 4548, 3))

    print("start color")
    for step, (ms4, pan, fusion, gt_xy) in enumerate(all_data_loader):
        ms4 = ms4.to(device)
        pan = pan.to(device)
        fusion = fusion.to(device)
        with torch.no_grad():
            output = model(ms4, pan, fusion)
        pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        # print("pred_y###",pred_y_numpy)
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [127, 255, 0]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 255]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [58, 58, 139]
            elif pred_y_numpy[k] == 7:
                class_count[7] = class_count[7] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [240, 32, 160]
            elif pred_y_numpy[k] == 8:
                class_count[8] = class_count[8] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [221, 160, 221]
            elif pred_y_numpy[k] == 9:
                class_count[9] = class_count[9] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
            elif pred_y_numpy[k] == 10:
                class_count[10] = class_count[10] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 255]
            elif pred_y_numpy[k] == 11:
                class_count[11] = class_count[11] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]

        if step % 1000 == 0:
            print("Now is ", step)
    print(class_count)
    cv2.imwrite("back_bone1.png", out_color)



