from typing import Dict
import numpy as np
import torch
from torch.nn import functional as F
import cv2

from Dataset.dataset_new import create_train_data_loader
# from Dataset.dataset import create_train_data_loader
from Diffusion.cls_model import cls_model
import os
import torch.optim as optim
from Scheduler import GradualWarmupScheduler
from Diffusion.kappa_AA import aa_oa

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train_cls(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # create data
    pan_path = modelConfig["pan_path"]
    ms_path = modelConfig["ms_path"]
    fusion_path = modelConfig["fusion_img_path"]
    # label_path = modelConfig["label_path"]
    train_label_path = modelConfig["train_label_path"]
    test_label_path = modelConfig["test_label_path"]
    Ms4_patch_size = modelConfig["Ms4_patch_size"]
    BATCH_SIZE = modelConfig["BATCH_SIZE"]
    Train_Rate = modelConfig["Train_Rate"]

    train_loader, test_loader = create_train_data_loader(pan_path, ms_path, fusion_path, train_label_path=train_label_path,
                                                                          test_label_path=test_label_path,
                                                                          Ms4_patch_size=Ms4_patch_size,
                                                                          BATCH_SIZE=BATCH_SIZE, Train_Rate=Train_Rate)
    # old dataset
    # train_loader, test_loader = create_train_data_loader(pan_path, ms_path, fusion_path,label_path,
    #                                                      Ms4_patch_size=Ms4_patch_size,
    #                                                      BATCH_SIZE=BATCH_SIZE, Train_Rate=Train_Rate)

    print("create cls model")
    # 此处的output是模型输出，多少类就是多少
    """
    nanjing = 11
    beijing = 10
    bigXian = 12
    """
    model = cls_model(output=12)

    optimizer = optim.Adam(model.parameters(), lr=modelConfig["cls_lr"])

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["cls_epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["cls_epoch"] // 10,
        after_scheduler=cosineScheduler)

    if modelConfig["cls_model_path"] is not None:
        model.load_state_dict(
            torch.load(modelConfig["cls_model_path"], map_location="cpu")
        )

    model.to(device)

    epoch = modelConfig["cls_epoch"]
    for i in range(epoch):
        model.train()
        correct = 0.0
        for step, (ms, pan, fusion, label, _) in enumerate(train_loader):
            ms, pan, fusion, label = ms.to(device), pan.to(device), fusion.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(ms, pan, fusion)

            pred_train = output.max(1, keepdim=True)[1]
            correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
            loss = F.cross_entropy(output, label.long())
            # 定义反向传播
            loss.backward()
            # 定义优化
            optimizer.step()
            if step % 1000 == 0:
                print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(i, loss.item(), step))
        print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))
        warmUpScheduler.step()
        if (i + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'cls_' + str(i+1) + "_" + "_.pt"))

    for i in range(1):
        model.eval()
        correct = 0.0
        test_loss = 0.0
        # test_matrix 大小为和类别一样
        test_matrix = np.zeros([12, 12])
        with torch.no_grad():
            for step, (ms, pan, fusion, label, _) in enumerate(test_loader):
                ms, pan, fusion, label = ms.to(device), pan.to(device), fusion.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(ms, pan, fusion)
                # output = model(fusion)
                test_loss += F.cross_entropy(output, label.long())
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred).long()).sum().item()

                for i in range(len(label)):
                    test_matrix[int(pred[i].item())][int(label[i].item())] += 1


            test_loss = test_loss / len(test_loader.dataset)

            print("test-average loss: {:.4f}, Accuracy(OA):{:.3f} \n".format(
                test_loss, 100.0 * correct / len(test_loader.dataset)
            ))

            AA_OA = aa_oa(test_matrix)
            print(test_matrix)





