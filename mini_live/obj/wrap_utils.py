index_wrap = [0, 2, 11, 12, 13, 14, 15, 16, 17, 18, 32, 36, 37, 38, 39, 40, 41, 42, 43, 50, 57, 58, 61,
              62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95,
              96, 97, 98, 100, 101, 106, 116, 117, 118, 119, 123, 129, 132, 135, 136, 137, 138, 140,
              142, 146, 147, 148, 149, 150, 152, 164, 165, 167, 169, 170, 171, 172, 175, 176, 177,
              178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 191, 192, 194, 199, 200, 201, 202,
              203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 227, 234, 262, 266,
              267, 268, 269, 270, 271, 272, 273, 280, 287, 288, 291, 292, 302, 303, 304, 306, 307,
              308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
              326, 327, 329, 330, 335, 345, 346, 347, 348, 352, 358, 361, 364, 365, 366, 367, 369,
              371, 375, 376, 377, 378, 379, 391, 393, 394, 395, 396, 397, 400, 401, 402, 403, 404,
              405, 406, 407, 408, 409, 410, 411, 415, 416, 418, 421, 422, 423, 424, 425, 426, 427,
              428, 430, 431, 432, 433, 434, 435, 436, 447, 454]

# index_edge_wrap = [111,43,57,21,76,59,68,67,78,66,69,168,177,169,170,161,176,123,159,145,208]
index_edge_wrap = [110,60,79,108,61,58,73,67,78,66,69,168,177,169,173,160,163,205,178,162,207]
index_edge_wrap_upper = [111, 110, 51, 52, 53, 54, 48, 63, 56, 47, 46, 1, 148, 149, 158, 165, 150, 156, 155, 154, 153, 207, 208]

print(len(index_wrap), len(set(index_wrap)))

# index_wrap = index_wrap + [291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
#                            146, 91, 181, 84, 17, 314, 405, 321, 375,]
import numpy as np
# 求平均人脸
def newWrapModel(wrapModel, face_pts_mean_personal_primer):

        face_wrap_entity = wrapModel.copy()
        # 正规点
        face_wrap_entity[:len(index_wrap),:3] = face_pts_mean_personal_primer[index_wrap, :3]
        # 边缘点
        vert_mid = face_wrap_entity[:,:3][index_edge_wrap[:4] + index_edge_wrap[-4:]].mean(axis=0)
        for index_, jj in enumerate(index_edge_wrap):
            face_wrap_entity[len(index_wrap) + index_,:3] = face_wrap_entity[jj, :3] + (face_wrap_entity[jj, :3] - vert_mid) * 0.32
            face_wrap_entity[len(index_wrap) + index_, 2] = face_wrap_entity[jj, 2]
        # 牙齿点
        from talkingface.utils import INDEX_LIPS, main_keypoints_index, INDEX_LIPS_UPPER, INDEX_LIPS_LOWER
        # 上嘴唇中点
        mid_upper_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS_UPPER], axis=0)
        mid_upper_teeth = np.mean(face_wrap_entity[-36:-18, :3], axis=0)
        tmp = mid_upper_teeth - mid_upper_mouth
        face_wrap_entity[-36:-18, :2] = face_wrap_entity[-36:-18, :2] - tmp[:2]
        # # 下嘴唇中点
        # mid_lower_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS_LOWER], axis=0)
        # mid_lower_teeth = np.mean(face_wrap_entity[-18:, :3], axis=0)
        # tmp = mid_lower_teeth - mid_lower_mouth
        # # print(tmp)
        # face_wrap_entity[-18:, :2] = face_wrap_entity[-18:, :2] - tmp[:2]

        return face_wrap_entity