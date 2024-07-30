from torch.optim import lr_scheduler

import torch.nn as nn
import torch

######################################################### training utils##########################################################

def get_scheduler(optimizer, niter,niter_decay,lr_policy='lambda',lr_decay_iters=50):
    '''
     scheduler in training stage
    '''
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch  - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

class GANLoss(nn.Module):
    '''
    GAN loss
    '''
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



import tqdm
import numpy as np
import cv2
import glob
import os
import math
import pickle
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
                      71,63,105,66,107,336,296,334,293,301,
                      168,197,5,4,75,97,2,326,305,
                      33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
def ExtractFaceFromFrameList(frames_list, vid_height, vid_width, out_size = 256):
    pts_3d = np.zeros([len(frames_list), 478, 3])
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        for index, frame in tqdm.tqdm(enumerate(frames_list)):
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print("****** WARNING! No face detected! ******")
                pts_3d[index] = 0
                return
                # continue
            image_height, image_width = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                for index_, i in enumerate(face_landmarks.landmark):
                    x_px = min(math.floor(i.x * image_width), image_width - 1)
                    y_px = min(math.floor(i.y * image_height), image_height - 1)
                    z_px = min(math.floor(i.z * image_height), image_height - 1)
                    pts_3d[index, index_] = np.array([x_px, y_px, z_px])

    # 计算整个视频中人脸的范围

    x_min, y_min, x_max, y_max = np.min(pts_3d[:, :, 0]), np.min(
        pts_3d[:, :, 1]), np.max(
        pts_3d[:, :, 0]), np.max(pts_3d[:, :, 1])
    new_w = int((x_max - x_min) * 0.55)*2
    new_h = int((y_max - y_min) * 0.6)*2
    center_x = int((x_max + x_min) / 2.)
    center_y = int(y_min + (y_max - y_min) * 0.6)
    size = max(new_h, new_w)
    x_min, y_min, x_max, y_max = int(center_x - size // 2), int(center_y - size // 2), int(
        center_x + size // 2), int(center_y + size // 2)

    # 确定裁剪区域上边top和左边left坐标
    top = y_min
    left = x_min
    # 裁剪区域与原图的重合区域
    top_coincidence = int(max(top, 0))
    bottom_coincidence = int(min(y_max, vid_height))
    left_coincidence = int(max(left, 0))
    right_coincidence = int(min(x_max, vid_width))

    scale = out_size / size
    pts_3d = (pts_3d - np.array([left, top, 0])) * scale
    pts_3d = pts_3d

    face_rect = np.array([center_x, center_y, size])
    print(np.array([x_min, y_min, x_max, y_max]))

    img_array = np.zeros([len(frames_list), out_size, out_size, 3], dtype = np.uint8)
    for index, frame in tqdm.tqdm(enumerate(frames_list)):
        img_new = np.zeros([size, size, 3], dtype=np.uint8)
        img_new[top_coincidence - top:bottom_coincidence - top, left_coincidence - left:right_coincidence - left,:] = \
            frame[top_coincidence:bottom_coincidence, left_coincidence:right_coincidence, :]
        img_new = cv2.resize(img_new, (out_size, out_size))
        img_array[index] = img_new
    return pts_3d,img_array, face_rect

