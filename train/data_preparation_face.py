import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
# sys.path.append("..")
import uuid
import tqdm
import numpy as np
import cv2
import glob
import math
import pickle
from talkingface.util.smooth import smooth_array
from talkingface.run_utils import calc_face_mat
import tqdm
from talkingface.utils import *
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8
def detect_face(frame):
    # 剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80的
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections or len(results.detections) > 1:
            return -1, None
        rect = results.detections[0].location_data.relative_bounding_box
        out_rect = [rect.xmin, rect.xmin + rect.width, rect.ymin, rect.ymin + rect.height]
        nose_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.NOSE_TIP)
        l_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.LEFT_EYE)
        r_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        # print(nose_, l_eye_, r_eye_)
        if nose_.x > l_eye_.x or nose_.x < r_eye_.x:
            return -2, out_rect

        h, w = frame.shape[:2]
        # print(frame.shape)
        if rect.xmin < 0 or rect.ymin < 0 or rect.xmin + rect.width > w or rect.ymin + rect.height > h:
            return -3, out_rect
        if rect.width * w < 100 or rect.height * h < 100:
            return -4, out_rect
    return 1, out_rect


def calc_face_interact(face0, face1):
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / ((x_max - x_min) * (y_max - y_min))
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / ((x_max - x_min) * (y_max - y_min))
    return min(tmp0, tmp1)


def detect_face_mesh(frame):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts_3d = np.zeros([478, 3])
        if not results.multi_face_landmarks:
            print("****** WARNING! No face detected! ******")
        else:
            image_height, image_width = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                for index_, i in enumerate(face_landmarks.landmark):
                    x_px = min(math.floor(i.x * image_width), image_width - 1)
                    y_px = min(math.floor(i.y * image_height), image_height - 1)
                    z_px = min(math.floor(i.z * image_width), image_width - 1)
                    pts_3d[index_] = np.array([x_px, y_px, z_px])
        return pts_3d


def ExtractFromVideo(video_path, circle = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    dir_path = os.path.dirname(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    totalFrames = int(totalFrames)
    pts_3d = np.zeros([totalFrames, 478, 3])
    face_rect_list = []

    # os.makedirs("../preparation/{}/image".format(model_name))
    for frame_index in tqdm.tqdm(range(totalFrames)):
        ret, frame = cap.read()  # 按帧读取视频
        # #到视频结尾时终止
        if ret is False:
            break
        # cv2.imwrite("../preparation/{}/image/{:0>6d}.png".format(model_name, frame_index), frame)
        tag_, rect = detect_face(frame)
        if frame_index == 0 and tag_ != 1:
            print("第一帧人脸检测异常，请剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80")
            pts_3d = -1
            break
        elif tag_ == -1:  # 有时候人脸检测会失败，就用上一帧的结果替代这一帧的结果
            rect = face_rect_list[-1]
        elif tag_ != 1:
            print("第{}帧人脸检测异常，请剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80, tag: {}".format(frame_index, tag_))
            # exit()
        if len(face_rect_list) > 0:
            face_area_inter = calc_face_interact(face_rect_list[-1], rect)
            # print(frame_index, face_area_inter)
            if face_area_inter < 0.6:
                print("人脸区域变化幅度太大，请复查，超出值为{}, frame_num: {}".format(face_area_inter, frame_index))
                pts_3d = -2
                break

        face_rect_list.append(rect)

        x_min = rect[0] * vid_width
        y_min = rect[2] * vid_height
        x_max = rect[1] * vid_width
        y_max = rect[3] * vid_height
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        # x_min = int(max(0, x_mid - seq_w * 0.65))
        # y_min = int(max(0, y_mid - seq_h * 0.4))
        # x_max = int(min(vid_width, x_mid + seq_w * 0.65))
        # y_max = int(min(vid_height, y_mid + seq_h * 0.8))
        crop_size = int(max(seq_w * 1.35, seq_h * 1.35))
        x_min = int(max(0, x_mid - crop_size * 0.5))
        y_min = int(max(0, y_mid - crop_size * 0.45))
        x_max = int(min(vid_width, x_min + crop_size))
        y_max = int(min(vid_height, y_min + crop_size))

        frame_face = frame[y_min:y_max, x_min:x_max]
        # cv2.imshow("s", frame_face)
        # cv2.waitKey(20)
        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])
    cap.release()  # 释放视频对象
    return pts_3d


def run(video_path, export_imgs = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    totalFrames = int(totalFrames)
    cap.release()
    pts_3d = ExtractFromVideo(video_path)
    if type(pts_3d) is np.ndarray and len(pts_3d) == totalFrames:
        print("关键点已提取")
    else:
        print("error in video: {}!!!".format(video_path))
        return

    video_name = os.path.basename(video_path).split(".")[0]
    video_data_path = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(video_data_path, exist_ok=True)

    if export_imgs:
        # 计算整个视频中人脸的范围
        x_min, y_min, x_max, y_max = np.min(pts_3d[:, :, 0]), np.min(
            pts_3d[:, :, 1]), np.max(
            pts_3d[:, :, 0]), np.max(pts_3d[:, :, 1])
        new_w = int((x_max - x_min) * 0.55) * 2
        new_h = int((y_max - y_min) * 0.6) * 2
        center_x = int((x_max + x_min) / 2.)
        center_y = int(y_min + (y_max - y_min) * 0.6)
        size = max(new_h, new_w)
        x_min, y_min, x_max, y_max = int(center_x - size // 2), int(center_y - size // 2), int(
            center_x + size // 2), int(center_y + size // 2)

        # 确定裁剪区域上边top和左边left坐标
        top = y_min
        left = x_min
        # 裁剪区域与原图的重合区域
        top_coincidence = int(max(top, 0))
        bottom_coincidence = int(min(y_max, vid_height))
        left_coincidence = int(max(left, 0))
        right_coincidence = int(min(x_max, vid_width))
        print("人脸活动范围：{}:{}, {}:{}".format(top_coincidence, bottom_coincidence, left_coincidence, right_coincidence))
        out_size = 512
        scale = 512. / size
        pts_3d = (pts_3d - np.array([left_coincidence, top_coincidence, 0])) * scale
        Path_output_pkl = "{}/keypoint_rotate.pkl".format(video_data_path)
        with open(Path_output_pkl, "wb") as f:
            pickle.dump(pts_3d, f)
        os.makedirs("{}/image".format(video_data_path), exist_ok=True)
        ffmpeg_cmd = "ffmpeg -i {} -vf crop={}:{}:{}:{},scale=512:512:flags=neighbor -loglevel quiet -y {}/image/%06d.png".format(
            video_path,
            right_coincidence - left_coincidence,
            bottom_coincidence - top_coincidence,
            left_coincidence,
            top_coincidence,
            video_data_path
        )
        os.system(ffmpeg_cmd)


    img_filelist = glob.glob("{}/image/*.png".format(video_data_path))
    img_filelist.sort()

    Path_output_pkl = "{}/keypoint_rotate.pkl".format(video_data_path)

    with open(Path_output_pkl, "rb") as f:
        images_info = pickle.load(f)[:, main_keypoints_index, :]
    pts_driven = images_info.reshape(len(images_info), -1)
    pts_driven = smooth_array(pts_driven).reshape(len(pts_driven), -1, 3)


    face_pts_mean = np.loadtxt(os.path.join(current_dir, "../data/face_pts_mean_mainKps.txt"))
    mat_list, pts_normalized_list, face_pts_mean_personal = calc_face_mat(pts_driven, face_pts_mean)
    pts_normalized_list = np.array(pts_normalized_list)
    # print(face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1])
    # print(np.max(pts_normalized_list[:,INDEX_FACE_OVAL[:10], 1], axis = 1))
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 1],
                                                             axis=0) + np.arange(5, 25, 2)
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0],
                                                             axis=0) - (9 - np.arange(0, 10))
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1],
                                                              axis=0) - np.arange(5, 25, 2) + 28
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0],
                                                              axis=0) + np.arange(0, 10)

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

    with open("{}/face_mat_mask.pkl".format(video_data_path), "wb") as f:
        pickle.dump([mat_list, face_pts_mean_personal], f)



def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 2:
        print("Usage: python data_preparation.py <data_dir>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    data_dir = sys.argv[1]
    print(f"Video dir is set to: {data_dir}")
    # data_dir = r"F:\C\AI\CV\88"
    video_files = glob.glob("{}/*.mp4".format(data_dir))
    for video_path in tqdm.tqdm(video_files):
        run(video_path)

if __name__ == "__main__":
    main()
