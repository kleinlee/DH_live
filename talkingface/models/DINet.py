import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d

def make_coordinate_grid_3d(spatial_size, type):
    '''
        generate 3D coordinate grid
    '''
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1,-1, 1).repeat(d,1, w)
    xx = x.view(1,1, -1).repeat(d,h, 1)
    zz = z.view(-1,1,1).repeat(1,h,w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed,zz

class ResBlock1d(nn.Module):
    '''
        basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features,out_features,1)
        self.norm1 = BatchNorm1d(in_features)
        self.norm2 = BatchNorm1d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class ResBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock1d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding,stride=2)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class SameBlock1d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class SameBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class AdaAT(nn.Module):
    '''
       AdaAT operator
    '''
    def __init__(self,  para_ch,feature_ch, cuda = True):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
                    nn.Linear(para_ch, feature_ch),
                    nn.Sigmoid()
                )
        self.rotation = nn.Sequential(
                nn.Linear(para_ch, feature_ch),
                nn.Tanh()
            )
        self.translation = nn.Sequential(
                nn.Linear(para_ch, 2 * feature_ch),
                nn.Tanh()
            )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map,para_code):
        batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159#
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        return trans_feature

class DINet_five_Ref(nn.Module):
    def __init__(self, source_channel,ref_channel, cuda = True):
        super(DINet_five_Ref, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel,32,kernel_size=7, padding=3),
            DownBlock2d(32, 64, kernel_size=3, padding=1),
            DownBlock2d(64, 128,kernel_size=3, padding=1)
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(384, 128, kernel_size=3, padding=1),
            # SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            # SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),

        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(128, 256, cuda)
        self.out_conv = nn.Sequential(
            SameBlock2d(384, 128, kernel_size=3, padding=1),
            UpBlock2d(128,128,kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
    def ref_input(self, ref_img):
        ## reference image encoder
        self.ref_in_feature = self.ref_in_conv(ref_img)
        # print(self.ref_in_feature.size(), self.ref_in_feature)
        ## use AdaAT do spatial deformation on reference feature maps
        self.ref_trans_feature0 = self.appearance_conv_list[0](self.ref_in_feature)
    def interface(self, source_img, source_prompt):
        self.source_img = torch.cat([source_img, source_prompt], dim=1)
        ## source image encoder
        source_in_feature = self.source_in_conv(self.source_img)
        # print(source_in_feature.size(), source_in_feature)

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature,self.ref_in_feature],1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(img_para.size(), img_para)
        ## concat alignment feature and audio feature
        trans_para = img_para

        ref_trans_feature = self.adaAT(self.ref_trans_feature0, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(ref_trans_feature.size(), ref_trans_feature)
        ## feature decoder
        merge_feature = torch.cat([source_in_feature,ref_trans_feature],1)
        # print(merge_feature.size(), merge_feature)
        out = self.out_conv(merge_feature)
        return out
    def forward(self, source_img, source_prompt, ref_img):
        self.ref_input(ref_img)
        out = self.interface(source_img, source_prompt)
        return out


# from torch import nn
# import time
# import torch
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_Generator = DINet_five_Ref(6, 30).to(device)
# torch.save(model_Generator.state_dict(), "DINet_five_Ref.pth")
#
# source_img = torch.zeros([1,3,256,256]).cuda()
# source_prompt = torch.zeros([1,3, 256,256]).cuda()
# ref_img = torch.zeros([1,30,256,256]).cuda()
#
# model_Generator.ref_input(ref_img)
# start_time = time.time()
# for i in range(2000):
#     print(i, time.time() - start_time)
#     out = model_Generator.interface(source_img, source_prompt)