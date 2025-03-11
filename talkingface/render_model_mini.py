import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import random
import glob
import torch
import numpy as np
import cv2

from talkingface.utils import draw_mouth_maps
from talkingface.models.DINet_mini import input_height,input_width
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
        # cv2.imshow("ss", ref_face_edge)
        # cv2.waitKey(-1)
        # cv2.imshow("ss", ref_img)
        # cv2.waitKey(-1)
        ref_face_edge = cv2.resize(ref_face_edge, (128, 128))
        ref_img = cv2.resize(ref_img, (128, 128))
        w_pad = int((128 - input_width) / 2)
        h_pad = int((128 - input_height) / 2)

        ref_img = np.concatenate(
            [ref_img[h_pad:-h_pad, w_pad:-w_pad, :3], ref_face_edge[h_pad:-h_pad, w_pad:-w_pad, :1]], axis=2)
        # cv2.imshow("ss", ref_face_edge[h_pad:-h_pad, w_pad:-w_pad])
        # cv2.waitKey(-1)
        ref_img_list.append(ref_img)

        teeth_ref_img = os.path.join(current_dir, r"../video_data/teeth_ref/*.png")
        teeth_ref_img = random.sample(glob.glob(teeth_ref_img), 1)[0]
        teeth_ref_img = teeth_ref_img.replace("_2", "")
        teeth_ref_img = cv2.imread(teeth_ref_img, cv2.IMREAD_UNCHANGED)
        ref_img_list.append(teeth_ref_img)
        ref_img_list.append(teeth_ref_img)

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