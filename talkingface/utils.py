import numpy as np
import random
import cv2

INDEX_LEFT_EYEBROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
INDEX_RIGHT_EYEBROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
INDEX_EYEBROW = INDEX_LEFT_EYEBROW + INDEX_RIGHT_EYEBROW

INDEX_NOSE_EDGE = [343, 355, 358, 327, 326, 2, 97, 98, 129, 126, 114]
INDEX_NOSE_MID = [6, 197,195,5,4]
INDEX_NOSE = INDEX_NOSE_EDGE + INDEX_NOSE_MID

INDEX_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,308,415,310,311,312,13,82,81,80,191]
INDEX_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,409,270,269,267,0,37,39,40,185,]
INDEX_LIPS = INDEX_LIPS_INNER + INDEX_LIPS_OUTER

INDEX_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
INDEX_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
INDEX_EYE = INDEX_LEFT_EYE + INDEX_RIGHT_EYE
# 下半边脸的轮廓
INDEX_FACE_OVAL = [
    454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150,
    136, 172, 58, 132, 93, 234,
    # 206, 426
]

INDEX_MUSCLE = [
    371,266,425,427,434,394,
    169,214,207,205,36,142
]

main_keypoints_index = INDEX_EYEBROW + INDEX_NOSE + INDEX_LIPS + INDEX_EYE + INDEX_FACE_OVAL + INDEX_MUSCLE

rotate_ref_index = INDEX_EYEBROW + INDEX_NOSE + INDEX_EYE + INDEX_FACE_OVAL + INDEX_MUSCLE
# print(len(main_keypoints_index))
Normalized = True
if Normalized:
    tmp = 0
    list_ = []
    for i in [INDEX_LEFT_EYEBROW, INDEX_RIGHT_EYEBROW, INDEX_NOSE_EDGE, INDEX_NOSE_MID, INDEX_LIPS_INNER, INDEX_LIPS_OUTER, INDEX_LEFT_EYE, INDEX_RIGHT_EYE, INDEX_FACE_OVAL, INDEX_MUSCLE]:
        i = (tmp + np.arange(len(i))).tolist()
        list_.append(i)
        tmp += len(i)
[INDEX_LEFT_EYEBROW, INDEX_RIGHT_EYEBROW, INDEX_NOSE_EDGE, INDEX_NOSE_MID, INDEX_LIPS_INNER, INDEX_LIPS_OUTER, INDEX_LEFT_EYE, INDEX_RIGHT_EYE, INDEX_FACE_OVAL, INDEX_MUSCLE] = list_
INDEX_EYEBROW = INDEX_LEFT_EYEBROW + INDEX_RIGHT_EYEBROW
INDEX_NOSE = INDEX_NOSE_EDGE + INDEX_NOSE_MID
INDEX_LIPS = INDEX_LIPS_INNER + INDEX_LIPS_OUTER
INDEX_EYE = INDEX_LEFT_EYE + INDEX_RIGHT_EYE

INDEX_LIPS_LOWER = INDEX_LIPS_INNER[:11] + INDEX_LIPS_OUTER[:11][::-1]
INDEX_LIPS_UPPER = INDEX_LIPS_INNER[10:] + [INDEX_LIPS_INNER[0], INDEX_LIPS_OUTER[0]] + INDEX_LIPS_OUTER[10:][::-1]

FACE_MASK_INDEX = INDEX_FACE_OVAL[2:-2]
def crop_face(keypoints, is_train = False, size = [512, 512]):
    """
    x_ratio: 裁剪出一个正方形，边长根据keypoints的宽度 * x_ratio决定
    """
    x_min, y_min, x_max, y_max = np.min(keypoints[FACE_MASK_INDEX, 0]), np.min(keypoints[FACE_MASK_INDEX, 1]), np.max(keypoints[FACE_MASK_INDEX, 0]), np.max(
        keypoints[FACE_MASK_INDEX, 1])
    y_min = keypoints[33, 1]          # 两眼间的点开始y轴裁剪
    border_width_half = max(x_max - x_min, y_max - y_min) * 0.6
    center_x = int((x_min + x_max) / 2.0)
    center_y = int((y_min + y_max) / 2.0)
    if is_train:
        w_offset = random.randint(-2, 2)
        h_offset = random.randint(-2, 2)
        center_x = center_x + w_offset
        center_y = center_y + h_offset
    x_min, y_min, x_max, y_max = int(center_x - border_width_half), int(center_y - border_width_half), int(
        center_x + border_width_half), int(center_y + border_width_half)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(size[1], x_max)
    y_max = min(size[0], y_max)
    return [x_min, y_min, x_max, y_max]

def crop_mouth(pts_array_origin, img_w, img_h, is_train = False):
    center_x = np.mean(pts_array_origin[INDEX_LIPS_OUTER, 0])
    center_y = np.mean(pts_array_origin[INDEX_LIPS_OUTER, 1])
    x_min, y_min, x_max, y_max = np.min(pts_array_origin[INDEX_FACE_OVAL[2:-2], 0]), np.min(
        pts_array_origin[INDEX_FACE_OVAL[2:-2], 1]), np.max(
        pts_array_origin[INDEX_FACE_OVAL[2:-2], 0]), np.max(pts_array_origin[INDEX_FACE_OVAL[2:-2], 1])
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(x_max, img_w)
    y_max = min(y_max, img_h)
    new_size = max((x_max - x_min), (y_max - y_min))*0.46

    if is_train:
        h_offset = int(new_size * 0.04)
        h_offset = random.randint(-h_offset, h_offset)
        center_y = center_y + h_offset

    x_min, y_min, x_max, y_max = int(center_x - new_size), int(center_y - new_size*0.89), int(
        center_x + new_size), int(center_y + new_size*1.11)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    return np.array([x_min, y_min, x_max, y_max])

def draw_mouth_maps(keypoints, size=(256, 256), im_edges = None):
    w, h = size
    # edge map for face region from keypoints
    if im_edges is None:
        im_edges = np.zeros((h, w, 3), np.uint8)  # edge map for all edges
    pts = keypoints[INDEX_LIPS_OUTER, :2]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(im_edges, [pts], color=(0, 0, 0))

    pts = keypoints[INDEX_LIPS_UPPER, :2]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(im_edges, [pts], color=(255, 0, 0))
    pts = keypoints[INDEX_LIPS_LOWER, :2]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(im_edges, [pts], color=(127, 0, 0))
    return im_edges

def draw_face_feature_maps(keypoints, mode = ["mouth", "nose", "eye", "oval"], size=(256, 256), im_edges = None,  mouth_width = None, mouth_height = None):
    w, h = size
    # edge map for face region from keypoints
    if im_edges is None:
        im_edges = np.zeros((h, w, 3), np.uint8)  # edge map for all edges

    if "mouth_bias" in mode:

        w0, w1, h0, h1 = (int(keypoints[INDEX_NOSE_EDGE[5], 0] - mouth_width / 2),
                          int(keypoints[INDEX_NOSE_EDGE[5], 0] + mouth_width / 2),
                          int(keypoints[INDEX_NOSE_EDGE[5], 1] + mouth_height / 4),
                          int(keypoints[INDEX_NOSE_EDGE[5], 1] + mouth_height / 4 + mouth_height))
        w0, h0 = max(0, w0), max(h0, 0)
        # print(w0, w1, h0, h1)
        mouth_mask = np.zeros((h, w, 3), np.uint8)  # edge map for all edges
        mouth_mask[h0:h1, w0:w1] = 255
        mouth_index = np.where(mouth_mask == 255)
        blur = (10, 10)
        img_mouth = cv2.cvtColor(im_edges, cv2.COLOR_BGR2GRAY)
        img_mouth = cv2.blur(img_mouth, blur)
        # print(mouth_index)
        mean_ = int(np.mean(img_mouth[(mouth_index[0], mouth_index[1])]))
        max_, min_ = random.randint(mean_ + 40, mean_ + 70), random.randint(mean_ - 70, mean_ - 40)
        img_mouth = (img_mouth.astype(np.float32) - min_) / (max_ - min_) * 255.
        img_mouth = img_mouth.clip(0, 255).astype(np.uint8)
        # print(img_mouth.shape[0], img_mouth.shape[1])
        img_mouth = cv2.resize(img_mouth, (100, 50))

        # 定义噪声的标准差
        sigma = 8  # 你可以根据需要调整这个值
        # 生成与图片相同大小和类型的噪声
        noise = sigma * np.random.randn(img_mouth.shape[0], img_mouth.shape[1])
        # 将噪声添加到图片上
        img_mouth = img_mouth + noise
        img_mouth = cv2.resize(img_mouth.clip(0, 255).astype(np.uint8), (im_edges.shape[0], im_edges.shape[1]))

        img_mouth = np.concatenate(
            [img_mouth[:, :, np.newaxis], img_mouth[:, :, np.newaxis], img_mouth[:, :, np.newaxis]], axis=2)

        # bias_x = int(min(size[0] - max(mouth_index[0]), min(mouth_index[0])))
        # bias_y = int(min(size[0] - max(mouth_index[1]), min(mouth_index[1])))
        # print(bias_x, bias_y)
        # bias_x = random.randint(-bias_x // 10, bias_x // 10)
        # bias_y = random.randint(-bias_y // 10, bias_y // 10)
        #
        # mouth_index2 = np.array(mouth_index)
        # # print(mouth_index)
        # mouth_index2[0] = mouth_index[0] + bias_x
        # mouth_index2[1] = mouth_index[1] + bias_y
        # mouth_index2 = (mouth_index2[0], mouth_index2[1], mouth_index2[2])
        # # print(mouth_index2.T, mouth_index2.shape)
        #
        output = np.zeros((h, w, 3), np.uint8)
        # output[mouth_index2] = img_mouth[mouth_index]
        # return im_edges
        output[mouth_index] = img_mouth[mouth_index]
        im_edges = output
    if "nose" in mode:
        for ii in range(len(INDEX_NOSE_EDGE) - 1):
            pt1 = [int(flt) for flt in keypoints[INDEX_NOSE_EDGE[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_NOSE_EDGE[ii+1]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
        for ii in range(len(INDEX_NOSE_MID) -1):
            pt1 = [int(flt) for flt in keypoints[INDEX_NOSE_MID[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_NOSE_MID[ii + 1]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
    if "eye" in mode:
        for ii in range(len(INDEX_LEFT_EYE)):
            pt1 = [int(flt) for flt in keypoints[INDEX_LEFT_EYE[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_LEFT_EYE[(ii + 1)%len(INDEX_LEFT_EYE)]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
        for ii in range(len(INDEX_RIGHT_EYE)):
            pt1 = [int(flt) for flt in keypoints[INDEX_RIGHT_EYE[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_RIGHT_EYE[(ii + 1) % len(INDEX_RIGHT_EYE)]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
    if "oval" in mode:
        tmp =  INDEX_FACE_OVAL[:6]
        for ii in range(len(tmp) -1):
            pt1 = [int(flt) for flt in keypoints[tmp[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[tmp[ii + 1]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 0, 255), 2)
        tmp = INDEX_FACE_OVAL[-6:]
        for ii in range(len(tmp) - 1):
            pt1 = [int(flt) for flt in keypoints[tmp[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[tmp[ii + 1]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 0, 255), 2)

    # if "mouth_outer" in mode:
    #     pts = keypoints[INDEX_LIPS_OUTER]
    #     pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    #     cv2.fillPoly(im_edges, [pts], color=(255, 0, 0))
    # if "mouth" in mode:
    #     pts = keypoints[INDEX_LIPS_OUTER][:,2]
    #     pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    #     cv2.fillPoly(im_edges, [pts], color=(255, 0, 0))
    #     pts = keypoints[INDEX_LIPS_INNER][:,2]
    #     pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    #     cv2.fillPoly(im_edges, [pts], color=(0, 0, 0))
    # if "mouth_outer" in mode:
    #     for ii in range(len(INDEX_LIPS_OUTER)):
    #         pt1 = [int(flt) for flt in keypoints[INDEX_LIPS_OUTER[ii]]][:2]
    #         pt2 = [int(flt) for flt in keypoints[INDEX_LIPS_OUTER[(ii + 1)%len(INDEX_LIPS_OUTER)]]][:2]
    #         cv2.line(im_edges, tuple(pt1), tuple(pt2), (255, 0, 0), 2)



    if "mouth" in mode:
        for ii in range(len(INDEX_LIPS_OUTER)):
            pt1 = [int(flt) for flt in keypoints[INDEX_LIPS_OUTER[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_LIPS_OUTER[(ii + 1)%len(INDEX_LIPS_OUTER)]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
        for ii in range(len(INDEX_LIPS_INNER)):
            pt1 = [int(flt) for flt in keypoints[INDEX_LIPS_INNER[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_LIPS_INNER[(ii + 1)%len(INDEX_LIPS_INNER)]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
    if "muscle" in mode:
        for ii in range(len(INDEX_MUSCLE) - 1):
            pt1 = [int(flt) for flt in keypoints[INDEX_MUSCLE[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[INDEX_MUSCLE[(ii + 1) % len(INDEX_MUSCLE)]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (255, 255, 255), 2)

    if "oval_all" in mode:
        tmp =  INDEX_FACE_OVAL
        for ii in range(len(tmp) -1):
            pt1 = [int(flt) for flt in keypoints[tmp[ii]]][:2]
            pt2 = [int(flt) for flt in keypoints[tmp[ii + 1]]][:2]
            cv2.line(im_edges, tuple(pt1), tuple(pt2), (0, 0, 255), 2)

    return im_edges

def smooth_array(array, weight = [0.1,0.8,0.1], mode = "numpy"):
    '''

    Args:
        array: [n_frames, n_values]， 需要转换为[n_values, 1, n_frames]
        weight: Conv1d.weight, 一维卷积核权重
    Returns:
        array: [n_frames, n_values]， 光滑后的array
    '''
    if mode == "torch":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        input = torch.Tensor(np.transpose(array[:, np.newaxis, :], (2, 1, 0)))
        smooth_length = len(weight)
        assert smooth_length % 2 == 1, "卷积核权重个数必须使用奇数"
        pad = (smooth_length // 2, smooth_length // 2)  # 当pad只有两个参数时，仅改变最后一个维度, 左边扩充1列，右边扩充1列
        input = F.pad(input, pad, "replicate")

        with torch.no_grad():
            conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=smooth_length)
            # 卷积核的元素值初始化
            weight = torch.tensor(weight).view(1, 1, -1)
            conv1.weight = torch.nn.Parameter(weight)
            nn.init.constant_(conv1.bias, 0)  # 偏置值为0
            # print(conv1.weight)
            out = conv1(input)
        return out.permute(2, 1, 0).squeeze().numpy()
    else:
        # out = np.zeros([array.shape[0] + 2, array.shape[1]])
        # input = np.zeros([array.shape[0] + 2, array.shape[1]])
        # input[0] = array[0]
        # input[-1] = array[-1]
        # input[1:-1] = array
        # for i in range(out.shape[1]):
        #     out[:, i] = np.convolve(input[:, i], weight, mode="same")
        # out0 = out[1:-1]
        smooth_length = len(weight)
        assert smooth_length % 2 == 1, "卷积核权重个数必须使用奇数"
        pad = smooth_length // 2
        fliter = np.array([weight]).T
        x0 = array
        fliter = np.repeat(fliter, x0.shape[1], axis=1)
        out0 = np.zeros_like(x0)
        for i in range(len(x0)):
            if i < pad or i >= len(x0) - pad:
                out0[i] = x0[i]
            else:
                tmp = x0[i - pad:i + pad + 1] * fliter
                out0[i] = np.sum(tmp, axis=0)
        return out0

def generate_face_mask():
    face_mask = np.zeros([256, 256], dtype=np.uint8)
    for i in range(20):
        ii = 19 - i
        face_mask[ii, :] = 13 * i
        face_mask[255 - ii, :] = 13 * i
        face_mask[:, ii] = 13 * i
        face_mask[:, 255 - ii] = 13 * i
    face_mask = np.array([face_mask, face_mask, face_mask]).transpose(1, 2, 0).astype(float) / 255.
    print(face_mask.shape)
    return face_mask

from math import cos,sin,radians
def RotateAngle2Matrix(tmp):   #tmp为xyz的旋转角,角度值
    tmp = [radians(i) for i in tmp]
    matX = np.array([[1.0,          0,            0],
                     [0.0,          cos(tmp[0]), -sin(tmp[0])],
                     [0.0,          sin(tmp[0]),  cos(tmp[0])]])
    matY = np.array([[cos(tmp[1]),  0,            sin(tmp[1])],
                     [0.0, 1, 0],
                     [-sin(tmp[1]),  0,            cos(tmp[1])]])
    matZ = np.array([[cos(tmp[2]), -sin(tmp[2]),  0],
                     [sin(tmp[2]),  cos(tmp[2]),  0],
                     [0, 0, 1]])
    matRotate = np.matmul(matZ, matY)
    matRotate = np.matmul(matRotate, matX)
    return matRotate


def normalizeLips(face_pts, face_pts_mean):
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
    # 九组对应点的距离
    bias_mouth = []
    for i in range(9):
        index0 = INDEX_MP_LIPS[60:80][i+1]
        index1 = INDEX_MP_LIPS[60:80][-i]
        bias_mouth.append(np.linalg.norm(face_pts_mean[index0] - face_pts_mean[index1]) - np.linalg.norm(face_pts[index0] - face_pts[index1]))

    # 分别应用这九组点的偏差
    for i in range(9):
        for j in range(4):
            index0 = INDEX_MP_LIPS[j*20:j*20 + 20][i + 1]
            index1 = INDEX_MP_LIPS[j*20:j*20 + 20][-i]
            face_pts[index0, 1] -= bias_mouth[i] / 2
            face_pts[index1, 1] += bias_mouth[i] / 2
    return face_pts