import torch
from torch import nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F
import cv2
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

input_width = 72
input_height = 72

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=2)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True,
                 sample_mode='nearest'):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu
        self.sample_mode = sample_mode

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode=self.sample_mode)
        out = self.conv(out)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv2(out)
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
        self.cuda = cuda
        self.f_dim = (20, input_height//4, input_width//4)
        if cuda:
            self.grid_xy, self.grid_z = make_coordinate_grid_3d(self.f_dim, torch.cuda.FloatTensor)
        else:
            self.grid_xy, self.grid_z = make_coordinate_grid_3d(self.f_dim, torch.FloatTensor)
            batch = 1
            self.grid_xy = self.grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            self.grid_z = self.grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)


    def forward(self, feature_map,para_code):
        # batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        # batch= feature_map.size(0)
        if self.cuda:
            batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
            # print(batch, d, h, w)
            grid_xy = self.grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1).view(batch, d, h*w, 2)
            grid_z = self.grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        else:
            batch = 1
            d, h, w = self.f_dim
            grid_xy = self.grid_xy.view(batch, d, h*w, 2)
            grid_z = self.grid_z
        # print((d, h, w), feature_map.type())
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        # print(scale.size(), scale)
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159#
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        # rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        # grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        # grid_xy = self.grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        # grid_z = self.grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).repeat(1, 1, h*w, 1)
        # print(scale.size(), scale)
        # rotation_matrix = rotation_matrix.unsqueeze(2).repeat(1, 1, h*w, 1, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).repeat(1, 1, h * w, 1)  # torch.Size([bs, 256, 4096, 4])
        translation = translation.unsqueeze(2).repeat(1, 1, h*w, 1)

        # trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        trans_grid = torch.matmul(rotation_matrix.view(batch, d, h * w, 2, 2), grid_xy.unsqueeze(-1))

        trans_grid = trans_grid.squeeze(-1) * scale + translation
        # print(trans_grid.view(batch, d, h, w, 2).size(), grid_z.unsqueeze(-1).size())
        # trans_grid = torch.matmul(rotation_matrix.view(batch, d, h * w, 2, 2), grid_xy.unsqueeze(-1)).squeeze(
        #     -1) * scale + translation
        # print(trans_grid.view(batch, d, h, w, 2).size(), grid_z.unsqueeze(-1).size())
        full_grid = torch.cat([trans_grid.view(batch, d, h, w, 2), grid_z.unsqueeze(-1)], -1)
        # print(full_grid.size(), full_grid)

        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        # print("trans_feature", trans_feature.size())
        return trans_feature

class DINet_mini(nn.Module):
    def __init__(self, source_channel,ref_channel, cuda = True):
        super(DINet_mini, self).__init__()
        f_dim = 20
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 12, kernel_size=3, padding=1),
            DownBlock2d(12, 12, kernel_size=3, padding=1),
            DownBlock2d(12, f_dim, kernel_size=3, padding=1),
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 16, kernel_size=3, padding=1),
            DownBlock2d(16, 16, kernel_size=3, padding=1),
            SameBlock2d(16, 16, kernel_size=3, padding=1),
            DownBlock2d(16, f_dim, kernel_size=3, padding=1),
            # DownBlock2d(ref_channel, 1, kernel_size=3, padding=1),
            # DownBlock2d(1, f_dim, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 16 →8
            DownBlock2d(f_dim*2, f_dim, kernel_size=3, padding=1),
            # 8 →4
            DownBlock2d(f_dim, f_dim, kernel_size=3, padding=1),
        )

        self.appearance_conv = nn.Sequential(
                ResBlock2d(f_dim, f_dim, 3, 1),
            )
        self.adaAT = AdaAT(f_dim, f_dim, cuda)
        self.out_conv = nn.Sequential(
            SameBlock2d(f_dim*2, f_dim, kernel_size=3, padding=1),
            UpBlock2d(f_dim, f_dim, kernel_size=3, padding=1),
            SameBlock2d(f_dim, 16, 3, 1),
            UpBlock2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)

    def ref_input(self, ref_img):
        ## reference image encoder
        self.ref_img = ref_img
        self.ref_in_feature = self.ref_in_conv(self.ref_img)
        # import pickle
        # with open("xxx.pkl", "wb") as f:
        #     pickle.dump(self.ref_in_feature, f)
        # print("ref_in_feature", self.ref_in_feature.size())

    def interface(self, source_img):
        self.source_img = source_img
        ## source image encoder
        source_in_feature = self.source_in_conv(self.source_img)
        img_para = self.trans_conv(torch.cat([source_in_feature, self.ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)

        ref_trans_feature = self.adaAT(self.ref_in_feature, img_para)
        ref_trans_feature = self.appearance_conv(ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv(merge_feature)
        return out

    def forward(self, ref_img, driving_img):
        self.driving_img = driving_img
        self.ref_input(ref_img)
        out = self.interface(self.driving_img)
        return out

class DINet_mini_pipeline(nn.Module):
    def __init__(self, source_channel,ref_channel, cuda = True):
        super(DINet_mini_pipeline, self).__init__()
        self.infer_model = DINet_mini(source_channel,ref_channel, cuda = cuda)

        self.grid_tensor = F.affine_grid(torch.eye(2, 3).unsqueeze(0).float(), (1, 1, 128, 128), align_corners=False).to("cuda" if cuda else "cpu")

        face_fusion_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/face_fusion_mask.png"))
        face_fusion_tensor = torch.from_numpy(face_fusion_tensor[:,:,:1] / 255.).float().permute(2, 0, 1).unsqueeze(0)
        mouth_fusion_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/mouth_fusion_mask.png"))
        mouth_fusion_tensor = cv2.resize(mouth_fusion_tensor, (input_width, input_height))
        mouth_fusion_tensor = torch.from_numpy(mouth_fusion_tensor[:,:,:1] / 255.).float().permute(2, 0, 1).unsqueeze(0)

        self.face_fusion_tensor = face_fusion_tensor.to("cuda" if cuda else "cpu")
        self.mouth_fusion_tensor = mouth_fusion_tensor.to("cuda" if cuda else "cpu")

    def ref_input(self, ref_tensor):
        self.infer_model.ref_input(ref_tensor)

    def interface(self, source_tensor, gl_tensor):
        face_mask = F.relu(torch.abs(gl_tensor[:, 2:3] * 2 - 1) - 0.9)* 10
        # face_mask = (gl_tensor[:, 2] == 1) | ((gl_tensor[:, 2] == 0))
        # face_mask = face_mask.float()

        deformation = gl_tensor[:, :2] * 2 - 1
        # 假设deformation的前两通道分别代表X和Y方向的位移
        # deformation的形状为[1, 2, H, W]，而grid的形状为[1, H, W, 2]
        deformation = deformation * self.face_fusion_tensor
        deformation = deformation.permute(0, 2, 3, 1)

        warped_grid = self.grid_tensor - deformation

        # img0应该是一个形状为[1, C, H, W]的tensor
        warped_img0 = F.grid_sample(source_tensor, warped_grid, mode='bilinear', padding_mode='zeros',
                                    align_corners=False)

        # # print(warped_img0.size(), face_mask.size())
        warped_tensor = warped_img0[:, :3]*(1-face_mask)
        # # warped_tensor = F.interpolate(warped_tensor, (128, 128))
        # w_pad = int((128 - 72) / 2)
        # h_pad = int((128 - 56) / 2)
        # gl_mouth_tensor = gl_tensor * face_mask
        # gl_mouth_tensor = gl_mouth_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]
        # fake_mouth_tensor = warped_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]
        # # fake_out = self.infer_model.interface(fake_mouth_tensor)
        # fake_mouth_tensor_input = fake_mouth_tensor*(1-self.mouth_fusion_tensor) + gl_mouth_tensor
        # fake_out = self.infer_model.interface(fake_mouth_tensor_input)
        # # fake_out = fake_mouth_tensor
        # fake_out = fake_out * self.mouth_fusion_tensor + fake_mouth_tensor*(1-self.mouth_fusion_tensor)
        # warped_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad] = fake_out
        # return warped_tensor

        # warped_tensor = warped_img0
        w_pad = int((128 - input_width) / 2)
        h_pad = int((128 - input_height) / 2)
        gl_mouth_tensor = gl_tensor * face_mask
        gl_mouth_tensor = gl_mouth_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]

        warped_mouth_tensor = warped_tensor
        warped_mouth_tensor = warped_mouth_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]

        fake_mouth_tensor_input = warped_mouth_tensor + gl_mouth_tensor
        fake_out = self.infer_model.interface(fake_mouth_tensor_input)
        # fake_out = warped_mouth_tensor
        # print(fake_out.size(), self.mouth_fusion_tensor.size(), warped_mouth_tensor.size())
        fake_out = fake_out * self.mouth_fusion_tensor + warped_mouth_tensor*(1-self.mouth_fusion_tensor)
        warped_img0[:, :3, h_pad:-h_pad, w_pad:-w_pad] = fake_out
        warped_img0[:,3] = source_tensor[:,3]
        return warped_img0

    def forward(self, source_tensor, gl_tensor, ref_tensor):
        '''

        Args:
            source_tensor: [batch, 3, 128, 128]
            gl_tensor: [batch, 3, 128, 128]
            ref_tensor: [batch, 12, 128, 128]

        Returns:
            warped_tensor: [batch, 3, 128, 128]
        '''
        self.ref_input(ref_tensor)
        warped_tensor = self.interface(source_tensor, gl_tensor)
        return warped_tensor


if __name__ == "__main__":
    device = "cpu"
    import torch.nn.functional as F
    # size = (56, 72)  # h, w
    # model = DINet_mini(3, 4*3, cuda = device=="cuda")
    # model.eval()
    # model = model.to(device)
    # driving_img = torch.zeros([1, 3, size[0], size[1]]).to(device)
    # ref_img = torch.zeros([1, 4*3, size[0], size[1]]).to(device)
    # from thop import profile
    # from thop import clever_format
    #
    # flops, params = profile(model.to(device), inputs=(ref_img, driving_img))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)

    size = (128, 128)  # h, w
    model = DINet_mini_pipeline(3, 4*3, cuda = device=="cuda")
    model.eval()
    model = model.to(device)
    source_tensor = torch.ones([1, 4, size[0], size[1]]).to(device)
    gl_tensor = torch.ones([1, 4, size[0], size[1]]).to(device)
    ref_tensor = torch.ones([1, 4*3, input_height, input_width]).to(device)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model.to(device), inputs=(source_tensor, gl_tensor, ref_tensor))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    import cv2

    out = model(source_tensor, gl_tensor, ref_tensor)
    image_numpy = out.detach().squeeze(0).cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    cv2.imwrite("sss1.png", image_numpy)

