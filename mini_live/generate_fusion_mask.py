import numpy as np
import cv2
import os

face_fusion_mask = np.zeros([128, 128], dtype = np.uint8)
for i in range(8):
    face_fusion_mask[i:-i,i:-i] = min(255, i*40)

cv2.imwrite("face_fusion_mask.png", face_fusion_mask)


from mini_live.obj.wrap_utils import index_wrap
image2 = cv2.imread("bs_texture.png")
image3 = np.zeros([12, 256, 3], dtype=np.uint8)
image3[:, :len(index_wrap)] = image2[:, index_wrap]
cv2.imwrite("bs_texture_halfFace.png", image3)