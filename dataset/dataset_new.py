import numpy as np
import torch
from libtiff import TIFF
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


def load_data(pan_path, ms_path, fusion_path, train_label_path, test_label_path):
    # 读取图片——ms4
    ms4_tif = TIFF.open(ms_path, mode='r')
    ms4_np = ms4_tif.read_image()
    ms4_np = ms4_np.astype(np.float32) # 图片过大，修改精度float64 -> float32
    # print(ms4_np)
    print('原始ms4图的形状：', np.shape(ms4_np))

    pan_tif = TIFF.open(pan_path, mode='r')
    pan_np = pan_tif.read_image()
    pan_np = pan_np.astype(np.float32)
    # print(pan_np)
    print('原始pan图的形状;', np.shape(pan_np))

    fusion_tif = TIFF.open(fusion_path, mode='r')
    fusion_np = fusion_tif.read_image()
    fusion_np = fusion_np.astype(np.float32)
    print('融合图像的形状', np.shape(fusion_np))

    train_label_np = np.load(train_label_path)
    print('train_label数组形状：', np.shape(train_label_np))

    test_label_np = np.load(test_label_path)
    print('test_label数组形状：', np.shape(test_label_np))

    return ms4_np, pan_np, fusion_np, train_label_np, test_label_np



def create_train_data_loader(pan_path, ms_path, fusion_path, train_label_path, test_label_path, Ms4_patch_size, BATCH_SIZE, Train_Rate):
    ms4_np, pan_np, fusion_np, train_label_np, test_label_np = load_data(pan_path, ms_path, fusion_path, train_label_path, test_label_path)
    # ms4与pan图补零  (给图片加边框）
    Ms4_patch_size = Ms4_patch_size  # ms4截块的边长
    Interpolation = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REPLICATE： 进行复制的补零操作;
    # cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
    # cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
    # cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;

    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的ms4图的形状：', np.shape(ms4_np))


    Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的pan图的形状：', np.shape(pan_np))

    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    fusion_np = cv2.copyMakeBorder(fusion_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的融合图的形状：', np.shape(fusion_np))

    # 按类别比例拆分数据集
    # label_np=label_np.astype(np.uint8)
    # label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
    #
    # label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
    # print('类标：', label_element)
    # print('各类样本数：', element_count)
    # Categories_Number = len(label_element) - 1  # 数据的类别数
    # print('标注的类别数：', Categories_Number)
    # label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

    train_label_np = train_label_np - 1  
    train_label_element, train_element_count = np.unique(train_label_np, return_counts=True)
    print('训练样本类标：', train_label_element)
    print('训练集各类样本数：', train_element_count)
    Categories_Number = len(train_label_element) - 1
    print('训练集标注的类别数：', Categories_Number)
    label_row, label_column = np.shape(train_label_np)

    test_label_np = test_label_np - 1
    test_label_element, test_element_count = np.unique(test_label_np, return_counts=True)
    print('测试样本类标：', test_label_element)
    print('测试集各类样本数：', test_element_count)
    print('测试集标注的类别数：', Categories_Number)

    '''归一化图片'''
    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    # ground_xy = np.array([[]] * Categories_Number).tolist()  # [[],[],[],[],[],[],[]]  产生与类别数相等的空列表
    # ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column,
    #                                                                     2)  # [H*W, 2] 二维数组

    train_ground_xy = np.array([[]] * Categories_Number).tolist()
    test_ground_xy = np.array([[]] * Categories_Number).tolist()
    ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column,
                                                                        2)  # [H*W, 2] 二维数组


    for row in range(label_row):  # 训练集与测试集的大小一样，所以用一个row和colum即可
        for column in range(label_column):
            # demo 为 255， 西安数据集为65535
            if train_label_np[row][column] != 255:
                train_ground_xy[int(train_label_np[row][column])].append([row, column])  # 记录属于每个类别的位置集合
            if test_label_np[row][column] != 255:
                test_ground_xy[int(test_label_np[row][column])].append([row, column])

    # 标签内打乱
    for categories in range(Categories_Number): # train与test有同样的类别，所以一个即可
        train_ground_xy[categories] = np.array(train_ground_xy[categories])
        test_ground_xy[categories] = np.array(test_ground_xy[categories])

        train_shuffle_array = np.arange(0, len(train_ground_xy[categories]), 1) # 从0到len(ground_xy[categories])逐个取一个数
        test_shuffle_array = np.arange(0, len(test_ground_xy[categories]), 1) # 从0到len(ground_xy[categories])逐个取一个数

        np.random.shuffle(train_shuffle_array)
        np.random.shuffle(test_shuffle_array)

        train_ground_xy[categories] = train_ground_xy[categories][train_shuffle_array]
        test_ground_xy[categories] = test_ground_xy[categories][test_shuffle_array]



    ground_xy_train = []
    ground_xy_test = []
    label_train = []
    label_test = []

    for categories in range(Categories_Number):
        train_categories_number = len(train_ground_xy[categories])
        test_categories_number = len(test_ground_xy[categories])

        for i in range(train_categories_number):
            ground_xy_train.append(train_ground_xy[categories][i])
        for j in range(test_categories_number):
            ground_xy_test.append(test_ground_xy[categories][j])

        label_train = label_train + [categories for x in range(int(train_categories_number))]
        label_test = label_test + [categories for x in range(int(test_categories_number))]

    label_train = np.array(label_train)
    label_test = np.array(label_test)
    ground_xy_train = np.array(ground_xy_train)
    ground_xy_test = np.array(ground_xy_test)

    # 训练数据与测试数据，数据集内打乱
    shuffle_array = np.arange(0, len(label_test), 1)
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    ground_xy_test = ground_xy_test[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    ground_xy_train = ground_xy_train[shuffle_array]

    label_train = torch.from_numpy(label_train).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
    ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)

    """随机抽取一部分坐标作为训练集和测试集，少量样本即可"""
    label_test = label_test[:1000000]
    ground_xy_test = ground_xy_test[:1000000]
    print('训练样本数：', len(label_train))
    print('测试样本数：', len(label_test))

    # 数据归一化
    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    fusion = to_tensor(fusion_np)

    pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
    ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道
    fusion = np.array(fusion).transpose((2, 0, 1))

    # 转换类型
    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)
    fusion = torch.from_numpy(fusion).type(torch.FloatTensor)

    class MyData(Dataset):
        def __init__(self, MS4, Pan, fusion ,Label, xy, cut_size):
            self.train_data1 = MS4
            self.train_data2 = Pan
            self.train_data3 = fusion
            self.train_labels = Label
            self.gt_xy = xy
            self.cut_ms_size = cut_size
            self.cut_pan_size = cut_size * 4

        def __getitem__(self, index):
            x_ms, y_ms = self.gt_xy[index]
            x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
            y_pan = int(4 * y_ms)
            image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                       y_ms:y_ms + self.cut_ms_size]

            image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            image_fusion = self.train_data3[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            locate_xy = self.gt_xy[index]

            target = self.train_labels[index]
            """ms: 16x16 => 64x64"""
            image_ms = image_ms.unsqueeze(1)  # 增加一个维度
            image_ms = F.interpolate(image_ms, scale_factor=4, mode='bicubic', align_corners=False)
            image_ms = image_ms.squeeze(1)  # 删除一个维度

            # print("image.size", image_pan.shape)
            return image_ms, image_pan, image_fusion, target, locate_xy

        def __len__(self):
            return len(self.gt_xy)

    print("Creating train dataloader.")
    train_data = MyData(ms4, pan, fusion, label_train, ground_xy_train, Ms4_patch_size)
    print("Creating test dataloader.")
    test_data = MyData(ms4, pan, fusion, label_test, ground_xy_test, Ms4_patch_size)
    print("Creating all dataloader.")

    print("the training dataset is length:{}".format(len(train_data)))
    print("the test dataset is length:{}".format(len(test_data)))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=True, pin_memory=True)  # cpu
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader


def create_color_data_loader(pan_path, ms_path, fusion_path, label_path, Ms4_patch_size, BATCH_SIZE):
    ms4_np, pan_np, fusion_np, label_np = load_data(pan_path, ms_path, fusion_path, label_path)
    # ms4与pan图补零  (给图片加边框）
    Ms4_patch_size = Ms4_patch_size  # ms4截块的边长
    Interpolation = cv2.BORDER_REFLECT_101

    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的ms4图的形状：', np.shape(ms4_np))


    Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的pan图的形状：', np.shape(pan_np))

    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    fusion_np = cv2.copyMakeBorder(fusion_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的融合图的形状：', np.shape(fusion_np))

    # 按类别比例拆分数据集
    # label_np=label_np.astype(np.uint8)
    label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

    label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
    print('类标：', label_element)
    print('各类样本数：', element_count)
    Categories_Number = len(label_element) - 1  # 数据的类别数
    print('标注的类别数：', Categories_Number)
    label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

    '''归一化图片'''
    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    ground_xy = np.array([[]] * Categories_Number).tolist()  # [[],[],[],[],[],[],[]]  产生与类别数相等的空列表
    ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column,
                                                                        2)  # [H*W, 2] 二维数组

    count = 0
    for row in range(label_row):  # 行
        for column in range(label_column):
            ground_xy_allData[count] = [row, column]
            count = count + 1
            # demo 为 255， 西安数据集为65535
            if label_np[row][column] != 65535:
                ground_xy[int(label_np[row][column])].append([row, column])  # 记录属于每个类别的位置集合

    # 标签内打乱
    for categories in range(Categories_Number):
        ground_xy[categories] = np.array(ground_xy[categories])
        shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
        np.random.shuffle(shuffle_array)

        ground_xy[categories] = ground_xy[categories][shuffle_array]
    shuffle_array = np.arange(0, label_row * label_column, 1)
    np.random.shuffle(shuffle_array)
    ground_xy_allData = ground_xy_allData[shuffle_array]

    ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

    # 数据归一化
    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    fusion = to_tensor(fusion_np)

    pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
    ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道
    fusion = np.array(fusion).transpose((2, 0, 1))

    # 转换类型
    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)
    fusion = torch.from_numpy(fusion).type(torch.FloatTensor)


    class MyData1(Dataset):
        def __init__(self, MS4, Pan, fusion,xy, cut_size):
            self.train_data1 = MS4
            self.train_data2 = Pan
            self.train_data3 = fusion
            self.gt_xy = xy
            self.cut_ms_size = cut_size
            self.cut_pan_size = cut_size * 4

        def __getitem__(self, index):
            x_ms, y_ms = self.gt_xy[index]
            x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
            y_pan = int(4 * y_ms)
            image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                       y_ms:y_ms + self.cut_ms_size]

            image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            image_fusion = self.train_data3[:, x_pan:x_pan + self.cut_pan_size,
                           y_pan:y_pan + self.cut_pan_size]

            locate_xy = self.gt_xy[index]

            """ms: 16x16 => 64x64"""
            image_ms = image_ms.unsqueeze(1)  # 增加一个维度
            image_ms = F.interpolate(image_ms, scale_factor=4, mode='bicubic', align_corners=False)
            image_ms = image_ms.squeeze(1)  # 删除一个维度

            return image_ms, image_pan, image_fusion, locate_xy

        def __len__(self):
            return len(self.gt_xy)

    print("Creating all dataloader.")
    all_data = MyData1(ms4, pan, fusion, ground_xy_allData, Ms4_patch_size)
    all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return all_data_loader

