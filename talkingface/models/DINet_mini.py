import torch
from torch import nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F
import cv2
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

input_width = 104 + 8
input_height = 72 + 8
model_size = 168 + 16

class ImprovedGeneratorBlock(nn.Module):
    def __init__(self, in_features, out_features, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.adain = AdaptiveModulation(out_features, style_dim)

    def forward(self, x, style_vector):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.adain(x, style_vector)
        return F.relu(x, inplace=True)


# 1. 定义深度可分离卷积 (MobileNet 的核心组件)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        # 深度卷积：每个通道独立卷积，参数量和计算量极低
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        # 逐点卷积：1x1 卷积，负责通道间的信息融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdaptiveModulation(nn.Module):
    """
    保持不变，这是风格注入的核心
    """

    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        # 输出变为 3 份：scale, bias, mask
        # mask让模型根据当前运动特征 x 的形状，结合 style_vector指令，计算出“我应该把牙齿颜色涂在哪个具体位置”。
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * 3)
        )

    def forward(self, x, style_vector):
        normalized = self.norm(x)
        style_params = self.mlp(style_vector)
        scale, bias, mask_logits = torch.chunk(style_params, 3, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        mask = torch.sigmoid(mask_logits.unsqueeze(-1).unsqueeze(-1) * x)
        modulated = normalized * (1 + scale) + bias
        return mask * modulated + (1 - mask) * normalized


class MobileGeneratorBlock(nn.Module):
    """
    改进的生成块：使用 SeparableConv 替代标准 Conv
    """

    def __init__(self, in_features, out_features, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        # 使用深度可分离卷积，大幅降低计算量
        self.conv = SeparableConv2d(in_features, out_features)
        self.adain = AdaptiveModulation(out_features, style_dim)

    def forward(self, x, style_vector):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.adain(x, style_vector)
        return F.relu(x, inplace=True)


class DINet_mini(nn.Module):
    '''
    使用一张背景是原图，但带有嘴唇和牙齿部位的分割图片，以及三张RGBA四通道组成的参考图片集合。
    使用轻量化算法，完成原始图片生成
    '''
    def __init__(self, source_channel, ref_channel, style_dim=64):
        super().__init__()
        self.ref_bg_feature = None

        self.bg_pool_18 = nn.AvgPool2d(kernel_size=4, stride=4)  # 72 -> 18
        self.bg_pool_36 = nn.AvgPool2d(kernel_size=2, stride=2)  # 72 -> 36
        self.skip_fuse0 = nn.Conv2d(3, 48, kernel_size=1)
        self.skip_fuse1 = nn.Conv2d(3, 24, kernel_size=1)

        # 1. Motion Encoder (保持原有结构或简化)
        # 提取源图的几何结构 (128x128 -> 18x18)
        self.motion_encoder = nn.Sequential(
            SeparableConv2d(source_channel, 32, stride=2),
            nn.LeakyReLU(0.1, inplace=True),  # 建议统一改为 LeakyReLU 增加平滑度
            SeparableConv2d(32, 64, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            SeparableConv2d(64, style_dim, stride=2),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. Appearance Encoder (生成全局Style Vector)
        # 注意：不再需要复杂的Warp，只需要提取高层语义
        # self.appearance_encoder = nn.Sequential(
        #     # Layer 1: 提取低级特征 (边缘、简单纹理)
        #     nn.Conv2d(ref_channel, 1, kernel_size=3, stride=2, padding=1),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(1, style_dim)
        # )
        self.appearance_encoder = nn.Sequential(
            # Layer 1: 提取低级特征 (边缘、简单纹理)
            nn.Conv2d(ref_channel, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # Layer 2: 提取中级特征 (五官部件、光照方向)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # Layer 3: 提取高级语义特征 (全局肤色、风格化特征)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # 全局池化 + 压缩
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, style_dim, kernel_size=1, bias=True),
        )

        # 3. Decoder (使用 AdaptiveModulation)
        # 输入是 Motion 特征，Style 来自参考图
        self.decoder_blocks = nn.ModuleList([
            # 第一层：标准卷积 (保结构)
            ImprovedGeneratorBlock(style_dim, 96, style_dim, upsample=False),

            # 中间两层：深度可分离卷积 (省算力)
            MobileGeneratorBlock(96, 48, style_dim, upsample=True),
            MobileGeneratorBlock(48, 24, style_dim, upsample=True),

            # 最后一层：标准卷积 (保画质)
            ImprovedGeneratorBlock(24, 16, style_dim, upsample=True),
        ])
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def ref_input(self, ref_img):
        ## reference image encoder
        self.ref_in_feature = self.appearance_encoder(ref_img)  # [B, 64]
        self.ref_in_feature = self.ref_in_feature.squeeze(-1).squeeze(-1)

    def interface(self, source_img):
        motion_feat = self.motion_encoder(source_img)  # [B, 64, 9, 9]

        # 提取原图前3个通道的背景特征
        bg_feat = source_img[:, :3, :, :]
        bg_18 = self.bg_pool_18(bg_feat)
        bg_36 = self.bg_pool_36(bg_feat)

        x = motion_feat
        # 手动循环，注入背景
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, self.ref_in_feature)

            # 在对应尺度的上采样后，把背景特征加进去
            if i == 1:
                # 通道数对齐 (64对64)
                x = x + self.skip_fuse0(bg_18)
            elif i == 2:
                x = x + self.skip_fuse1(bg_36)

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

    def forward(self, source_img, ref_img):
        # source_img: 变形后的源图 [B, 3, 128, 128]
        # ref_img: 参考图 [B, 12, 128, 128]

        self.ref_input(ref_img)
        x = self.interface(source_img)
        return x


class DINet_mini_pipeline(nn.Module):
    def __init__(self, source_channel, ref_channel, cuda=True):
        super(DINet_mini_pipeline, self).__init__()
        self.infer_model = DINet_mini(source_channel, ref_channel)

        self.grid_tensor = F.affine_grid(torch.eye(2, 3).unsqueeze(0).float(), (1, 1, model_size, model_size),
                                         align_corners=False).to("cuda" if cuda else "cpu")

        face_fusion_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/face_fusion_mask.png"))
        face_fusion_tensor = cv2.resize(face_fusion_tensor, (model_size, model_size))
        face_fusion_tensor = torch.from_numpy(face_fusion_tensor[:, :, :1] / 255.).float().permute(2, 0, 1).unsqueeze(0)
        mouth_fusion_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/mouth_fusion_mask.png"))
        mouth_fusion_tensor = cv2.resize(mouth_fusion_tensor, (input_width, input_height))
        mouth_fusion_tensor = torch.from_numpy(mouth_fusion_tensor[:, :, :1] / 255.).float().permute(2, 0, 1).unsqueeze(
            0)

        self.face_fusion_tensor = face_fusion_tensor.to("cuda" if cuda else "cpu")
        self.mouth_fusion_tensor = mouth_fusion_tensor.to("cuda" if cuda else "cpu")

        self.logo_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/MatesX_logo.png"), cv2.IMREAD_UNCHANGED)
        self.logo_tensor = cv2.cvtColor(self.logo_tensor, cv2.COLOR_BGRA2RGBA)
        self.logo_tensor = torch.from_numpy(self.logo_tensor / 255.).float().permute(2, 0, 1).unsqueeze(0)
        self.logo_tensor = self.logo_tensor.to("cuda" if cuda else "cpu")

    def ref_input(self, ref_tensor):
        self.infer_model.ref_input(ref_tensor)

    def interface(self, source_tensor, gl_tensor):
        face_mask = F.relu(torch.abs(gl_tensor[:, 2:3] * 2 - 1) - 0.9) * 10
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
        warped_tensor = warped_img0[:, :3] * (1 - face_mask)

        # warped_tensor = warped_img0
        w_pad = int((model_size - input_width) / 2)
        h_pad = int((model_size - input_height) / 2)
        gl_mouth_tensor = gl_tensor * face_mask
        gl_mouth_tensor = gl_mouth_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]

        warped_mouth_tensor = warped_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]

        # image_numpy = warped_mouth_tensor.detach().squeeze(0).cpu().float().numpy()
        # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        # image_numpy = image_numpy.clip(0, 255)
        # image_numpy = image_numpy.astype(np.uint8)
        # cv2.imshow("ss", image_numpy)
        # cv2.waitKey(-1)

        fake_mouth_tensor_input = warped_mouth_tensor + gl_mouth_tensor
        fake_out = self.infer_model.interface(fake_mouth_tensor_input)
        # fake_out = warped_mouth_tensor
        # print(fake_out.size(), self.mouth_fusion_tensor.size(), warped_mouth_tensor.size())
        fake_out = fake_out * self.mouth_fusion_tensor + warped_mouth_tensor * (1 - self.mouth_fusion_tensor)
        warped_img0[:, :3, h_pad:-h_pad, w_pad:-w_pad] = fake_out

        warped_img0[:, 3] = F.relu(warped_img0[:, 3] + face_mask)
        return warped_img0

    def forward(self, source_tensor, gl_tensor, ref_tensor):
        '''

        Args:
            source_tensor: [batch, 4, model_size, model_size]
            gl_tensor: [batch, 4, model_size, model_size]
            style_vector: [batch, 64]

        Returns:
            warped_tensor: [batch, 4, model_size, model_size]
        '''
        self.ref_input(ref_tensor)
        warped_tensor = self.interface(source_tensor, gl_tensor)
        return warped_tensor

class DINet_mini_pipeline2(nn.Module):
    def __init__(self, source_channel, ref_channel, cuda=True):
        super(DINet_mini_pipeline2, self).__init__()
        self.infer_model = DINet_mini(source_channel, ref_channel)

        self.grid_tensor = F.affine_grid(torch.eye(2, 3).unsqueeze(0).float(), (1, 1, model_size, model_size),
                                         align_corners=False).to("cuda" if cuda else "cpu")

        face_fusion_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/face_fusion_mask.png"))
        face_fusion_tensor = cv2.resize(face_fusion_tensor, (model_size, model_size))
        face_fusion_tensor = torch.from_numpy(face_fusion_tensor[:, :, :1] / 255.).float().permute(2, 0, 1).unsqueeze(0)
        mouth_fusion_tensor = cv2.imread(os.path.join(current_dir, "../../mini_live/mouth_fusion_mask.png"))
        mouth_fusion_tensor = cv2.resize(mouth_fusion_tensor, (input_width, input_height))
        mouth_fusion_tensor = torch.from_numpy(mouth_fusion_tensor[:, :, :1] / 255.).float().permute(2, 0, 1).unsqueeze(
            0)

        self.face_fusion_tensor = face_fusion_tensor.to("cuda" if cuda else "cpu")
        self.mouth_fusion_tensor = mouth_fusion_tensor.to("cuda" if cuda else "cpu")

    def interface(self, source_tensor, gl_tensor):
        face_mask = F.relu(torch.abs(gl_tensor[:, 2:3] * 2 - 1) - 0.9) * 10

        deformation = gl_tensor[:, :2] * 2 - 1
        # 假设deformation的前两通道分别代表X和Y方向的位移
        # deformation的形状为[1, 2, H, W]，而grid的形状为[1, H, W, 2]
        deformation = deformation * self.face_fusion_tensor
        deformation = deformation.permute(0, 2, 3, 1)

        warped_grid = self.grid_tensor - deformation

        warped_img0 = F.grid_sample(source_tensor, warped_grid, mode='bilinear', padding_mode='zeros',
                                    align_corners=False)
        warped_tensor = warped_img0[:, :3] * (1 - face_mask)

        # warped_tensor = warped_img0
        w_pad = int((model_size - input_width) / 2)
        h_pad = int((model_size - input_height) / 2)
        gl_mouth_tensor = gl_tensor * face_mask
        gl_mouth_tensor = gl_mouth_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]

        warped_mouth_tensor = warped_tensor[:, :3, h_pad:-h_pad, w_pad:-w_pad]

        fake_mouth_tensor_input = warped_mouth_tensor + gl_mouth_tensor
        fake_out = self.infer_model.interface(fake_mouth_tensor_input)
        fake_out = fake_out * self.mouth_fusion_tensor + warped_mouth_tensor * (1 - self.mouth_fusion_tensor)
        warped_img0[:, :3, h_pad:-h_pad, w_pad:-w_pad] = fake_out
        warped_img0[:, 3] = 1
        return warped_img0

    def forward(self, source_tensor, gl_tensor, style_vector):
        '''

        Args:
            source_tensor: [batch, 4, model_size, model_size]
            gl_tensor: [batch, 4, model_size, model_size]
            style_vector: [batch, 64]

        Returns:
            warped_tensor: [batch, 4, model_size, model_size]
        '''
        self.infer_model.ref_in_feature = style_vector
        warped_tensor = self.interface(source_tensor, gl_tensor)
        return warped_tensor

if __name__ == "__main__":
    device = "cpu"
    import torch.nn.functional as F

    size = (model_size, model_size)  # h, w
    model = DINet_mini_pipeline2(3, 4 * 3, cuda=device == "cuda")
    model.eval()
    model = model.to(device)

    # model_out_path = 'model.pth'
    # states = {
    #     'state_dict': {'net_g': model.infer_model.state_dict()},
    # }
    # torch.save(states, model_out_path)

    source_tensor = torch.ones([1, 4, size[0], size[1]]).to(device)
    gl_tensor = torch.ones([1, 4, size[0], size[1]]).to(device)
    style_tensor = torch.ones([1, 64]).to(device)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model.to(device), inputs=(source_tensor, gl_tensor, style_tensor))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    # import cv2
    #
    # out = model(source_tensor, gl_tensor, style_tensor)
    # image_numpy = out.detach().squeeze(0).cpu().float().numpy()
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    # image_numpy = image_numpy.clip(0, 255)
    # image_numpy = image_numpy.astype(np.uint8)
    # cv2.imwrite("sss1.png", image_numpy)
