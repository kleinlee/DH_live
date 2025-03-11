import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import pickle
import cv2
import numpy as np
import random
import pandas as pd
import glob
import copy
import torch
from torch.utils.data import DataLoader
from talkingface.data.DHLive_mini_dataset import Few_Shot_Dataset,data_preparation
from talkingface.utils import *
from talkingface.model_utils import device
# video_list = glob.glob(r"E:\data\video\video\*.mp4")
# video_list = [os.path.basename(i).split(".")[0] for i in video_list]

df = pd.read_csv(r"F:\C\AI\CV\DH008_few_shot\DH0119_mouth64_48/imageVar2.csv")
video_list = df[df["imageVar"] > 265000]["name"].tolist()
video_list = [os.path.dirname(os.path.dirname(i)) for i in video_list]
print(len(video_list))
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8
video_list = video_list[105:125]

dict_info = data_preparation(video_list)
test_set = Few_Shot_Dataset(dict_info, is_train=True, n_ref = 3)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

def Tensor2img(tensor_, channel_index):
    frame = tensor_[channel_index:channel_index + 3, :, :].detach().squeeze(0).cpu().float().numpy()
    frame = np.transpose(frame, (1, 2, 0)) * 255.0
    frame = frame.clip(0, 255)
    return frame.astype(np.uint8)
size_ = 256
for iteration, batch in enumerate(testing_data_loader):
    # source_tensor, source_prompt_tensor, ref_tensor, ref_prompt_tensor, target_tensor = [iii.to(device) for iii in batch]
    source_tensor, ref_tensor, target_tensor = [iii.to(device) for iii in batch[:3]]
    print(source_tensor.size(), ref_tensor.size(), target_tensor.size(), batch[3][0])

    frame0 = Tensor2img(source_tensor[0], 0)
    frame1 = Tensor2img(ref_tensor[0], 0)
    frame2 = Tensor2img(ref_tensor[0], 1)
    frame3 = Tensor2img(ref_tensor[0], 4)
    frame4 = Tensor2img(ref_tensor[0], 5)
    frame5 = Tensor2img(target_tensor[0], 0)

    # cv2.imwrite("in0.png", frame0)
    # cv2.imwrite("in1.png", frame1)
    # cv2.imwrite("in2.png", frame2)
    # cv2.imwrite("in3.png", frame3)
    # cv2.imwrite("in4.png", frame4)
    # exit()


    frame = np.concatenate([frame0, frame1, frame2, frame3, frame4, frame5], axis=1)

    cv2.imshow("ss", frame[:, :, ::-1])
    # if iteration > 840:
    #     cv2.waitKey(-1)
    cv2.waitKey(-1)
    # break
cv2.destroyAllWindows()