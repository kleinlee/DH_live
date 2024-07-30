from talkingface.utils import *
import os
import pickle
import copy
def Tensor2img(tensor_, channel_index):
    frame = tensor_[channel_index:channel_index + 3, :, :].detach().squeeze(0).cpu().float().numpy()
    frame = np.transpose(frame, (1, 2, 0)) * 255.0
    frame = frame.clip(0, 255)
    return frame.astype(np.uint8)
INDEX_EYE_CORNER = [INDEX_LEFT_EYE[0], INDEX_LEFT_EYE[8], INDEX_RIGHT_EYE[0], INDEX_RIGHT_EYE[8]]
# INDEX_FACE_OVAL_UPPER = INDEX_FACE_OVAL[7:-7]
INDEX_FACE_OVAL_UPPER = INDEX_FACE_OVAL[:7] + INDEX_FACE_OVAL[-7:]
def calc_face_mat(pts_array_origin, face_pts_mean):
    '''

    :param pts_array_origin: mediapipe检测出的人脸关键点
    :return:
    '''
    face_pts_mean_rotatePts = face_pts_mean[INDEX_EYEBROW + INDEX_NOSE + INDEX_EYE_CORNER + INDEX_FACE_OVAL_UPPER]
    A = np.zeros([len(face_pts_mean_rotatePts) * 3, 12])
    for i in range(len(face_pts_mean_rotatePts)):
        A[3 * i + 0, 0:3] = face_pts_mean_rotatePts[i]
        A[3 * i + 0, 3] = 1
        A[3 * i + 1, 4:7] = face_pts_mean_rotatePts[i]
        A[3 * i + 1, 7] = 1
        A[3 * i + 2, 8:11] = face_pts_mean_rotatePts[i]
        A[3 * i + 2, 11] = 1
    A_inverse = np.linalg.pinv(A)
    pts_normalized_list = []
    mat_list = []
    for i in pts_array_origin:
        i_rotatePts = i[INDEX_EYEBROW + INDEX_NOSE + INDEX_EYE_CORNER + INDEX_FACE_OVAL_UPPER]
        B = i_rotatePts.flatten()
        x = A_inverse.dot(B)
        rotationMatrix = np.zeros([4, 4])
        rotationMatrix[:3, :] = x.reshape(3, 4)
        rotationMatrix[3, 3] = 1
        mat_list.append(rotationMatrix)

    for index_, i in enumerate(pts_array_origin):
        rotationMatrix = mat_list[index_]
        keypoints = np.ones([4, len(i)])
        keypoints[:3, :] = i.T
        keypoints_normalized = np.linalg.inv(rotationMatrix).dot(keypoints).T
        pts_normalized_list.append(keypoints_normalized[:, :3])

    # mouth_normalized_adjust = np.mean(np.array(pts_normalized_list)[:, INDEX_LIPS], axis=0)
    # print(mouth_normalized_adjust.shape)

    face_pts_mean_personal = np.mean(np.array(pts_normalized_list), axis=0)
    # print("face_pts_mean_personal", face_pts_mean_personal.shape)
    face_pts_mean_rotatePts2 = face_pts_mean_personal[
        INDEX_EYEBROW + INDEX_NOSE + INDEX_EYE_CORNER + INDEX_FACE_OVAL_UPPER]
    A2 = np.zeros([len(face_pts_mean_rotatePts2) * 3, 12])
    for i in range(len(face_pts_mean_rotatePts2)):
        A2[3 * i + 0, 0:3] = face_pts_mean_rotatePts2[i]
        A2[3 * i + 0, 3] = 1
        A2[3 * i + 1, 4:7] = face_pts_mean_rotatePts2[i]
        A2[3 * i + 1, 7] = 1
        A2[3 * i + 2, 8:11] = face_pts_mean_rotatePts2[i]
        A2[3 * i + 2, 11] = 1
    A2_inverse = np.linalg.pinv(A2)
    pts_normalized_list = []
    mat_list = []
    for i in pts_array_origin:
        i_rotatePts = i[INDEX_EYEBROW + INDEX_NOSE + INDEX_EYE_CORNER + INDEX_FACE_OVAL_UPPER]
        B = i_rotatePts.flatten()
        x = A2_inverse.dot(B)
        rotationMatrix = np.zeros([4, 4])
        rotationMatrix[:3, :] = x.reshape(3, 4)
        rotationMatrix[3, 3] = 1
        mat_list.append(rotationMatrix)

    mat_list = np.array(mat_list)
    # mat_list必须要平滑，注意是针对每个视频分别平滑
    sub_mat_list = mat_list
    smooth_array_ = sub_mat_list.reshape(-1, 16)
    smooth_array_ = smooth_array(smooth_array_)
    # print(smooth_array_, smooth_array_.shape)
    smooth_array_ = smooth_array_.reshape(-1, 4, 4)
    mat_list = smooth_array_
    mat_list = [hh for hh in mat_list]

    for index_,i in enumerate(pts_array_origin):
        rotationMatrix = mat_list[index_]
        keypoints = np.ones([4, len(i)])
        keypoints[:3, :] = i.T
        keypoints_normalized = np.linalg.inv(rotationMatrix).dot(keypoints).T
        pts_normalized_list.append(keypoints_normalized[:,:3])

    return mat_list,pts_normalized_list,face_pts_mean_personal
face_pts_mean = None
def video_pts_process(pts_array_origin):
    global face_pts_mean
    if face_pts_mean is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        face_pts_mean = np.loadtxt(os.path.join(current_dir, "../data/face_pts_mean_mainKps.txt"))
    # 先根据pts_array_origin计算出旋转矩阵、去除旋转后的人脸关键点、面部mask、
    mat_list, pts_normalized_list, face_pts_mean_personal = calc_face_mat(pts_array_origin, face_pts_mean)
    pts_normalized_list = np.array(pts_normalized_list)
    face_mask_pts_normalized = face_pts_mean_personal[INDEX_FACE_OVAL].copy()
    face_mask_pts_normalized[:10, 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 1],
                                                             axis=0) + np.arange(5, 25, 2)
    face_mask_pts_normalized[:10, 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0],
                                                             axis=0) - (9 - np.arange(0, 10))
    face_mask_pts_normalized[-10:, 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1],
                                                              axis=0) - np.arange(5, 25, 2) + 28
    face_mask_pts_normalized[-10:, 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0],
                                                              axis=0) + np.arange(0, 10)
    face_mask_pts_normalized[10, 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[10], 1], axis=0) + 25

    face_mask_pts = np.zeros([len(mat_list), len(face_mask_pts_normalized), 2])
    for index_ in range(len(mat_list)):
        rotationMatrix = mat_list[index_]

        keypoints = np.ones([4, len(face_mask_pts_normalized)])
        keypoints[:3, :] = face_mask_pts_normalized.T
        driving_mask = rotationMatrix.dot(keypoints).T
        face_mask_pts[index_] = driving_mask[:,:2]

    return mat_list, pts_normalized_list, face_pts_mean_personal, face_mask_pts

def mouth_replace(pts_array_origin, frames_num):
    '''

    :param pts_array_origin: mediapipe检测出的人脸关键点
    :return:
    '''
    if os.path.isfile("face_pts_mean_mainKps.txt"):
        face_pts_mean = np.loadtxt("face_pts_mean_mainKps.txt")
    else:
        face_pts_mean = np.loadtxt("data/face_pts_mean_mainKps.txt")
    mat_list,pts_normalized_list,face_pts_mean_personal = calc_face_mat(pts_array_origin, face_pts_mean)
    face_personal = face_pts_mean_personal.copy()
    pts_normalized_list = np.array(pts_normalized_list)
    # face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1] = np.max(pts_normalized_list[:,INDEX_FACE_OVAL[:10], 1], axis = 0) + 20
    # face_pts_mean_personal[INDEX_FACE_OVAL[:10], 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0], axis=0) + 10
    # face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1], axis=0) + 20
    # face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0], axis=0) - 10
    # face_pts_mean_personal[INDEX_FACE_OVAL[10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[10], 1], axis=0) + 20
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1] = np.max(pts_normalized_list[:,INDEX_FACE_OVAL[:10], 1], axis = 0) + np.arange(5,25,2)
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0], axis=0) - (9 - np.arange(0,10))
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1], axis=0) - np.arange(5,25,2) + 28
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0], axis=0) + np.arange(0,10)

    face_pts_mean_personal[INDEX_FACE_OVAL[10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[10], 1], axis=0) + 25

    face_pts_mean_personal = face_pts_mean_personal[INDEX_FACE_OVAL]
    face_mask_pts = np.zeros([len(mat_list), len(face_pts_mean_personal), 2])
    for index_ in range(len(mat_list)):
        rotationMatrix = mat_list[index_]

        keypoints = np.ones([4, len(face_pts_mean_personal)])
        keypoints[:3, :] = face_pts_mean_personal.T
        driving_mask = rotationMatrix.dot(keypoints).T
        face_mask_pts[index_] = driving_mask[:,:2]

    iteration = frames_num // len(pts_array_origin) + 1
    if iteration == 1:
        pass
    else:
        pts_array_origin2 = copy.deepcopy(pts_array_origin)
        mat_list2 = copy.deepcopy(mat_list)
        face_mask_pts2 = copy.deepcopy(face_mask_pts)
        for i in range(iteration - 1):
            if i % 2 == 0:
                pts_array_origin2 = np.concatenate(
                    [pts_array_origin2, pts_array_origin[::-1]], axis=0)
                mat_list2 += mat_list[::-1]
                face_mask_pts2 = np.concatenate(
                    [face_mask_pts2, face_mask_pts[::-1]], axis=0)
            else:
                pts_array_origin2 = np.concatenate(
                    [pts_array_origin2, pts_array_origin], axis=0)
                mat_list2 += mat_list
                face_mask_pts2 = np.concatenate(
                    [face_mask_pts2, face_mask_pts], axis=0)
        pts_array_origin = pts_array_origin2
        mat_list = mat_list2
        face_mask_pts = face_mask_pts2

    pts_array_origin, mat_list, face_mask_pts = pts_array_origin[:frames_num], mat_list[:frames_num], face_mask_pts[:frames_num]
    return pts_array_origin, mat_list, face_mask_pts, face_personal, pts_normalized_list


def concat_output_2binfile(mat_list, pts_3d, face_pts_mean_personal, face_mask_pts_normalized):
    face_stable_pts_2d = np.zeros([len(mat_list), len(INDEX_FACE_OVAL + INDEX_MUSCLE), 2])  # 法令纹和脸部外轮廓关键点
    face_mask_pts_2d = np.zeros([len(mat_list), face_mask_pts_normalized.shape[0], 2])
    for index_, i in enumerate(mat_list):
        rotationMatrix = i
        # 法令纹和脸部外轮廓关键点
        driving_mouth_pts = face_pts_mean_personal[INDEX_FACE_OVAL + INDEX_MUSCLE]
        keypoints = np.ones([4, len(driving_mouth_pts)])
        keypoints[:3, :] = driving_mouth_pts.T
        driving_mouth_pts = rotationMatrix.dot(keypoints).T
        face_stable_pts_2d[index_] = driving_mouth_pts[:, :2]

        # 脸部mask关键点
        driving_mouth_pts = face_mask_pts_normalized
        keypoints = np.ones([4, len(driving_mouth_pts)])
        keypoints[:3, :] = driving_mouth_pts.T
        driving_mouth_pts = rotationMatrix.dot(keypoints).T
        face_mask_pts_2d[index_] = driving_mouth_pts[:, :2]


    pts_2d_main = pts_3d[:, main_keypoints_index, :2].reshape(len(pts_3d), -1)
    smooth_array_ = np.array(mat_list).reshape(-1, 16) * 100
    face_mask_pts_2d = face_mask_pts_2d.reshape(len(face_mask_pts_2d), -1)
    face_stable_pts_2d = face_stable_pts_2d.reshape(len(face_stable_pts_2d), -1)

    output = np.concatenate([smooth_array_, pts_2d_main, face_mask_pts_2d, face_stable_pts_2d], axis=1).astype(np.float32)
    return output
from talkingface.data.few_shot_dataset import select_ref_index,get_ref_images_fromVideo
def prepare_video_data(video_path, Path_pkl, ref_img_index_list, ref_img = None,save_ref = None):
    with open(Path_pkl, "rb") as f:
        images_info = pickle.load(f)[:, main_keypoints_index, :]

    pts_driven = images_info.reshape(len(images_info), -1)
    pts_driven = smooth_array(pts_driven).reshape(len(pts_driven), -1, 3)
    cap_input = cv2.VideoCapture(video_path)
    if ref_img is not None:
        ref_img = cv2.imread(ref_img).reshape(256, -1, 256, 3).transpose(0, 2, 1, 3).reshape(256, 256, -1)
    else:
        if ref_img_index_list is None:
            ref_img_index_list = select_ref_index(pts_driven, n_ref=5, ratio=1 / 2.)
        ref_img = get_ref_images_fromVideo(cap_input, ref_img_index_list, pts_driven[:, :, :2])

    mat_list, pts_normalized_list, face_pts_mean_personal, face_mask_pts = video_pts_process(pts_driven)

    if save_ref is not None:
        h, w, c = ref_img.shape
        ref_img_ = ref_img.reshape(h, w, -1, 3).transpose(0, 2, 1, 3).reshape(h, -1, 3)
        # ref_path = "ref2.png"
        cv2.imwrite(save_ref, ref_img_)
    #     logger.info("参考图片已存至{}.".format(ref_path))

    return pts_driven, mat_list, pts_normalized_list, face_mask_pts, ref_img, cap_input