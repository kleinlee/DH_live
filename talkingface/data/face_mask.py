import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
import pickle
import cv2
import numpy as np
import os
import glob
from talkingface.util.smooth import smooth_array
from talkingface.run_utils import calc_face_mat
import tqdm
from talkingface.utils import *

path_ = r"../../../preparation_mix"
video_list = [os.path.join(path_, i) for i in os.listdir(path_)]
path_ = r"../../../preparation_hdtf"
video_list += [os.path.join(path_, i) for i in os.listdir(path_)]
path_ = r"../../../preparation_vfhq"
video_list += [os.path.join(path_, i) for i in os.listdir(path_)]
path_ = r"../../../preparation_bilibili"
video_list += [os.path.join(path_, i) for i in os.listdir(path_)]
print(video_list)
video_list = video_list[:]
img_all = []
keypoints_all = []
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8
for path_ in tqdm.tqdm(video_list):
    img_filelist = glob.glob("{}/image/*.png".format(path_))
    img_filelist.sort()
    if len(img_filelist) == 0:
        continue
    img_all.append(img_filelist)

    Path_output_pkl = "{}/keypoint_rotate.pkl".format(path_)

    with open(Path_output_pkl, "rb") as f:
        images_info = pickle.load(f)[:, main_keypoints_index, :]
    pts_driven = images_info.reshape(len(images_info), -1)
    pts_driven = smooth_array(pts_driven).reshape(len(pts_driven), -1, 3)

    face_pts_mean = np.loadtxt(r"data\face_pts_mean_mainKps.txt")
    mat_list,pts_normalized_list,face_pts_mean_personal = calc_face_mat(pts_driven, face_pts_mean)
    pts_normalized_list = np.array(pts_normalized_list)
    # print(face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1])
    # print(np.max(pts_normalized_list[:,INDEX_FACE_OVAL[:10], 1], axis = 1))
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1] = np.max(pts_normalized_list[:,INDEX_FACE_OVAL[:10], 1], axis = 0) + np.arange(5,25,2)
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0], axis=0) - (9 - np.arange(0,10))
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1], axis=0) - np.arange(5,25,2) + 28
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0], axis=0) + np.arange(0,10)

    face_pts_mean_personal[INDEX_FACE_OVAL[10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[10], 1], axis=0) + 25

    # for keypoints_normalized in pts_normalized_list:
    #     img = np.zeros([1000,1000,3], dtype=np.uint8)
    #     for coor in face_pts_mean_personal:
    #         # coor = (coor +1 )/2.
    #         cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, (255, 0, 0), thickness)
    #     for coor in keypoints_normalized:
    #         # coor = (coor +1 )/2.
    #         cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
    #     cv2.imshow("a", img)
    #     cv2.waitKey(30)

    with open("{}/face_mat_mask20240722.pkl".format(path_), "wb") as f:
        pickle.dump([mat_list, face_pts_mean_personal], f)
