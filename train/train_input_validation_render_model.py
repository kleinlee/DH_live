import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
import pickle
import cv2
import numpy as np
import random

import glob
import sys
import torch
from torch.utils.data import DataLoader
from talkingface.data.few_shot_dataset import Few_Shot_Dataset,data_preparation
from talkingface.utils import *

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 2:
        print("Usage: python train_input_validation_render_model.py <data_dir>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    data_dir = sys.argv[1]
    # data_dir = r"F:\C\AI\CV\88"
    print(f"Video dir is set to: {data_dir}")

    video_list = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    dict_info = data_preparation(video_list)
    device = torch.device("cuda:0")
    test_set = Few_Shot_Dataset(dict_info, is_train=True, n_ref=1)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    def Tensor2img(tensor_, channel_index):
        frame = tensor_[channel_index:channel_index + 3, :, :].detach().squeeze(0).cpu().float().numpy()
        frame = np.transpose(frame, (1, 2, 0)) * 255.0
        frame = frame.clip(0, 255)
        return frame.astype(np.uint8)

    for iteration, batch in enumerate(testing_data_loader):
        # source_tensor, source_prompt_tensor, ref_tensor, ref_prompt_tensor, target_tensor = [iii.to(device) for iii in batch]
        source_tensor, ref_tensor, target_tensor = [iii.to(device) for iii in batch]
        print(source_tensor.size(), ref_tensor.size(), target_tensor.size())

        frame0 = Tensor2img(source_tensor[0], 0)
        frame1 = Tensor2img(source_tensor[0], 3)
        frame2 = Tensor2img(ref_tensor[0], 0)
        frame3 = Tensor2img(ref_tensor[0], 3)
        frame4 = Tensor2img(target_tensor[0], 0)
        frame = np.concatenate([frame0, frame1, frame2, frame3, frame4], axis=1)

        cv2.imshow("ss", frame)
        cv2.waitKey(-1)
        # break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()