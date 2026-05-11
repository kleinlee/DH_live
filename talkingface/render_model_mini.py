import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import random
import glob
import torch
import numpy as np
import cv2

from talkingface.utils import draw_mouth_maps
from talkingface.models.DINet_mini import input_height,input_width,model_size
from talkingface.model_utils import device
class RenderModel_Mini:
    def __init__(self):
        self.__net = None

    def loadModel(self, ckpt_path):
        from talkingface.models.DINet_mini import DINet_mini_pipeline as DINet
        n_ref = 3
        source_channel = 3
        ref_channel = n_ref * 4
        self.net = DINet(source_channel, ref_channel, device == "cuda").to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        net_g_static = checkpoint['state_dict']['net_g']
        self.net.infer_model.load_state_dict(net_g_static)
        self.net.eval()


    def reset_charactor(self, ref_img, ref_keypoints, standard_size = 256):
        ref_img_list = []
        ref_face_edge = draw_mouth_maps(ref_keypoints, size=(standard_size, standard_size))
        ref_face_edge = cv2.resize(ref_face_edge, (model_size, model_size))
        ref_img = cv2.resize(ref_img, (model_size, model_size))
        w_pad = int((model_size - input_width) / 2)
        h_pad = int((model_size - input_height) / 2)

        h, w = ref_img.shape[:2]
        grid_h = h // 4
        grid_w = w // 4

        bg_feature = np.zeros(16, dtype=np.float32)
        for i in range(4):
            for j in range(4):
                # 提取当前网格区域
                grid_region = ref_img[i * grid_h: (i + 1) * grid_h, j * grid_w: (j + 1) * grid_w, :3]
                luminance = (0.299 * grid_region[:, :, 2].astype(np.float32) +
                             0.587 * grid_region[:, :, 1].astype(np.float32) +
                             0.114 * grid_region[:, :, 0].astype(np.float32))

                bg_feature[i * 4 + j] = np.clip(np.round(np.mean(luminance)), 0, 255)

        self.net.ref_bg_feature = bg_feature/255.

        ref_img = np.concatenate(
            [ref_img[h_pad:-h_pad, w_pad:-w_pad, :3], ref_face_edge[h_pad:-h_pad, w_pad:-w_pad, :1]], axis=2)
        # cv2.imshow("ss", ref_face_edge[h_pad:-h_pad, w_pad:-w_pad])
        # cv2.waitKey(-1)
        ref_img_list.append(ref_img)

        teeth_ref_img = os.path.join(current_dir, r"../video_data/teeth_ref/*_0.png")
        teeth_ref_img = random.sample(glob.glob(teeth_ref_img), 1)[0]
        # teeth_ref_img = teeth_ref_img.replace("_2", "")
        teeth_ref_img = cv2.imread(teeth_ref_img, cv2.IMREAD_UNCHANGED)
        teeth_ref_img = cv2.resize(teeth_ref_img, (input_width, input_height))
        teeth_ref_img = cv2.cvtColor(teeth_ref_img, cv2.COLOR_BGRA2RGBA)
        ref_img_list.append(teeth_ref_img)
        ref_img_list.append(teeth_ref_img)
        # tmp = np.concatenate(ref_img_list, axis = 1)
        # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGBA2BGRA)
        # cv2.imwrite("interface_rgb.png", tmp[:,:,:3])
        # cv2.imwrite("interface_rgba.png", tmp)
        # cv2.imshow("ss", tmp)
        # cv2.waitKey(-1)

        self.ref_img_save = np.concatenate([i[:,:,:3] for i in ref_img_list], axis=1)
        self.ref_img = np.concatenate(ref_img_list, axis=2)

        ref_tensor = torch.from_numpy(self.ref_img / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)

        self.net.ref_input(ref_tensor)


    def interface(self, source_tensor, gl_tensor):
        '''

        Args:
            source_tensor: [batch, 3, 128, 128]
            gl_tensor: [batch, 3, 128, 128]

        Returns:
            warped_img: [batch, 3, 128, 128]
        '''
        warped_img = self.net.interface(source_tensor, gl_tensor)
        return warped_img

    def save(self, path):
        torch.save(self.net.state_dict(), path)