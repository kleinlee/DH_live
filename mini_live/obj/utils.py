import uuid

import numpy as np
import cv2
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

def translation_matrix(point):
    """生成平移矩阵"""
    return np.array([
        [1, 0, 0, point[0]],
        [0, 1, 0, point[1]],
        [0, 0, 1, point[2]],
        [0, 0, 0, 1]
    ])
def rotate_around_point(point, theta, phi, psi):
    """围绕点P旋转"""
    # 将点P平移到原点
    T1 = translation_matrix(-point)

    # 定义欧拉角
    theta = np.radians(theta)  # 俯仰角
    phi = np.radians(phi)  # 偏航角
    psi = np.radians(psi)  # 翻滚角

    # 创建旋转矩阵
    tmp = [theta, phi, psi]
    matX = np.array([[1.0,            0,               0,               0],
                     [0.0,            np.cos(tmp[0]), -np.sin(tmp[0]),  0],
                     [0.0,            np.sin(tmp[0]),  np.cos(tmp[0]),  0],
                     [0,              0,               0,               1]
                     ])
    matY = np.array([[np.cos(tmp[1]), 0,               np.sin(tmp[1]),  0],
                     [0,              1,               0,               0],
                     [-np.sin(tmp[1]),0,               np.cos(tmp[1]),  0],
                     [0,              0,               0,               1]
                     ])
    matZ = np.array([[np.cos(tmp[2]), -np.sin(tmp[2]), 0,               0],
                     [np.sin(tmp[2]), np.cos(tmp[2]),  0,               0],
                     [0,              0,               1,               0],
                     [0,              0,               0,               1]
                     ])

    R = matZ @ matY @ matX

    # 将点P移回其原始位置
    T2 = translation_matrix(point)

    # 总的变换矩阵
    total_transform = T2 @ R @ T1

    return total_transform

def rodrigues_rotation_formula(axis, theta):
    """Calculate the rotation matrix using Rodrigues' rotation formula."""
    axis = np.asarray(axis) / np.linalg.norm(axis)  # Normalize the axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    return R
def RotateAngle2Matrix(center, axis, theta):
    """Rotate around a center point."""
    # Step 1: Translate the center to the origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center

    # Step 2: Apply the rotation
    R = rodrigues_rotation_formula(axis, theta)
    R_ = np.eye(4)
    R_[:3,:3] = R
    R = R_

    # Step 3: Translate back to the original position
    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    # Combine the transformations
    rotation_matrix = translation_back @ R @ translation_to_origin

    return rotation_matrix

INDEX_FLAME_LIPS = [
1,26,23,21,8,155,83,96,98,101,
73,112,123,124,143,146,71,52,51,40,
2,25,24,22,7,156,82,97,99,100,
74,113,122,125,138,148,66,53,50,41,
30,31,32,38,39,157,111,110,106,105,
104,120,121,126,137,147,65,54,49,48,
4,28,33,20,19,153,94,95,107,103,
76,118,119,127,136,149,64,55,47,46,

3,27,35,17,18,154,93,92,109,102,
75,114,115,128,133,151,61,56,43,42,
6,29, 13, 12, 11, 158, 86, 87, 88, 79,
80,117, 116, 135, 134, 150, 62, 63, 44, 45,
5,36,14,9,10,159,85,84,89,78,
77,141,130,131,139,145,67,59,58,69,
0,37,34,15,16,152,91,90,108,81,72,
142,129,132,140,144,68,60,57,70,
]
INDEX_MP_LIPS = [
291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
146, 91, 181, 84, 17, 314, 405, 321, 375,
306, 408, 304, 303, 302, 11, 72, 73, 74, 184, 76,
77, 90, 180, 85, 16, 315, 404, 320, 307,
292, 407, 272, 271, 268, 12, 38, 41, 42, 183, 62,
96, 89, 179, 86, 15, 316, 403, 319, 325,
308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
95, 88, 178, 87, 14, 317, 402, 318, 324,
]

def crop_mouth(mouth_pts, mat_list__):
    """
    x_ratio: 裁剪出一个正方形，边长根据keypoints的宽度 * x_ratio决定
    """
    num_ = len(mouth_pts)
    keypoints = np.ones([4, num_])
    keypoints[:3, :] = mouth_pts.T
    keypoints = mat_list__.dot(keypoints).T
    keypoints = keypoints[:, :3]

    x_min, y_min, x_max, y_max = np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1])
    border_width_half = max(x_max - x_min, y_max - y_min) * 0.66
    y_min = y_min + border_width_half * 0.3
    center_x = (x_min + x_max) /2.
    center_y = (y_min + y_max) /2.
    x_min, y_min, x_max, y_max = int(center_x - border_width_half), int(center_y - border_width_half*0.75), int(
        center_x + border_width_half), int(center_y + border_width_half*0.75)
    print([x_min, y_min, x_max, y_max])

    # pts = np.array([
    #     [x_min, y_min],
    #     [x_max, y_min],
    #     [x_max, y_max],
    #     [x_min, y_max]
    # ])
    return [x_min, y_min, x_max, y_max]

def drawMouth(keypoints, source_texture, out_size = (700, 1400)):
    INDEX_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    INDEX_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, ]
    INDEX_LIPS_LOWWER = INDEX_LIPS_INNER[:11] + INDEX_LIPS_OUTER[:11][::-1]
    INDEX_LIPS_UPPER = INDEX_LIPS_INNER[10:] + [INDEX_LIPS_INNER[0], INDEX_LIPS_OUTER[0]] + INDEX_LIPS_OUTER[10:][::-1]
    INDEX_LIPS = INDEX_LIPS_INNER + INDEX_LIPS_OUTER
    # keypoints = keypoints[INDEX_LIPS]
    keypoints[:, 0] = keypoints[:, 0] * out_size[0]
    keypoints[:, 1] = keypoints[:, 1] * out_size[1]
    # pts = keypoints[20:40]
    # pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    # cv2.fillPoly(source_texture, [pts], color=(255, 0, 0,))
    # pts = keypoints[:20]
    # pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    # cv2.fillPoly(source_texture, [pts], color=(0, 0, 0,))

    pts = keypoints[INDEX_LIPS_OUTER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_texture, [pts], color=(0, 0, 0))
    pts = keypoints[INDEX_LIPS_UPPER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_texture, [pts], color=(255, 0, 0))
    pts = keypoints[INDEX_LIPS_LOWWER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_texture, [pts], color=(127, 0, 0))

    prompt_texture = np.zeros_like(source_texture)
    pts = keypoints[INDEX_LIPS_UPPER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(prompt_texture, [pts], color=(255, 0, 0))
    pts = keypoints[INDEX_LIPS_LOWWER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(prompt_texture, [pts], color=(127, 0, 0))
    return source_texture, prompt_texture



#
# def draw_face_feature_maps(keypoints, mode = ["mouth", "nose", "eye", "oval"], size=(256, 256), im_edges = None):
#     w, h = size
#     # edge map for face region from keypoints
#     if im_edges is None:
#         im_edges = np.zeros((h, w, 3), np.uint8)  # edge map for all edges
#     if "mouth" in mode:
#         pts = keypoints[INDEX_LIPS_OUTER]
#         pts = pts.reshape((-1, 1, 2)).astype(np.int32)
#         cv2.fillPoly(im_edges, [pts], color=(0, 0, 0))
#         pts = keypoints[INDEX_LIPS_UPPER]
#         pts = pts.reshape((-1, 1, 2)).astype(np.int32)
#         cv2.fillPoly(im_edges, [pts], color=(255, 0, 0))
#         pts = keypoints[INDEX_LIPS_LOWWER]
#         pts = pts.reshape((-1, 1, 2)).astype(np.int32)
#         cv2.fillPoly(im_edges, [pts], color=(127, 0, 0))
#     return im_edges