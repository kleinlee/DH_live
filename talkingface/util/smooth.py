import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_array(array, weight = [0.1,0.8,0.1]):
    '''

    Args:
        array: [n_frames, n_values]， 需要转换为[n_values, 1, n_frames]
        weight: Conv1d.weight, 一维卷积核权重
    Returns:
        array: [n_frames, n_values]， 光滑后的array
    '''
    input = torch.Tensor(np.transpose(array[:,np.newaxis,:], (2, 1, 0)))
    smooth_length = len(weight)
    assert smooth_length%2 == 1, "卷积核权重个数必须使用奇数"
    pad = (smooth_length//2, smooth_length//2)    # 当pad只有两个参数时，仅改变最后一个维度, 左边扩充1列，右边扩充1列
    input = F.pad(input, pad, "replicate")

    with torch.no_grad():
        conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=smooth_length)
        # 卷积核的元素值初始化
        weight = torch.tensor(weight).view(1, 1, -1)
        conv1.weight = torch.nn.Parameter(weight)
        nn.init.constant_(conv1.bias, 0)  # 偏置值为0
        # print(conv1.weight)
        out = conv1(input)
    return out.permute(2,1,0).squeeze().numpy()

if __name__ == '__main__':
    model_id = "new_case"
    Path_output_pkl = "../preparation/{}/mouth_info.pkl".format(model_id + "/00001")
    import pickle
    with open(Path_output_pkl, "rb") as f:
        images_info = pickle.load(f)
    pts_array_normalized = np.array(images_info[2])
    pts_array_normalized = pts_array_normalized.reshape(-1, 16)
    smooth_array_ = smooth_array(pts_array_normalized)
    print(smooth_array_, smooth_array_.shape)
    smooth_array_ = smooth_array_.reshape(-1, 4, 4)
    import pandas as pd

    pd.DataFrame(smooth_array_[:, :, 0]).to_csv("mat2.csv")