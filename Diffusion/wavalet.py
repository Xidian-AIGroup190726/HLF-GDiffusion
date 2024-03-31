import torch
import torch.fft as fft


def bandpass_filter(image, lowcut, highcut):
    # 获取图像的二维傅立叶变换
    f_transform = fft.fft2(image)
    f_transform_shift = fft.fftshift(f_transform)

    # 创建一个掩模，其中低频部分和高频部分被过滤掉\

    # rows, cols = image.shape
    rows = image.shape[2]
    cols = image.shape[3]
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros((rows, cols), dtype=torch.complex64).to('cuda')
    mask[crow - highcut:crow + highcut, ccol - highcut:ccol + highcut] = 1
    mask[crow - lowcut:crow + lowcut, ccol - lowcut:ccol + lowcut] = 0

    # 应用掩模
    f_transform_shift *= mask

    # 获取傅立叶逆变换
    f_transform_ishift = fft.ifftshift(f_transform_shift)
    image_filtered = fft.ifft2(f_transform_ishift)
    image_filtered = abs(image_filtered)

    return image_filtered




