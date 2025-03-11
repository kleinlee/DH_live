import uuid
import tqdm
import numpy as np
import cv2
import sys
import os
import math
import pickle
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def detect_face(frame, min_detection_confidence = 0.5):
    # 剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80的
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_detection_confidence) as face_detection:
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
        if out_rect[0] < 0 or out_rect[2] < 0 or out_rect[1] > w or out_rect[3] > h:
            return -3, out_rect
        if rect.width * w < 60 or rect.height * h < 60:
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


def ExtractFromVideo(video_path, face_rect=None):
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

        if frame_index == 0:
            # 检测人脸
            tag_, rect = detect_face(frame, min_detection_confidence = 0.25)
            if tag_ != 1:
                tag_, rect = detect_face(frame[int(0.1 * vid_height):int(0.9 * vid_height),
                                         int(0.1 * vid_width):int(0.9 * vid_width)], min_detection_confidence=0.25)
                assert tag_ == 1, "第一帧检测不到人脸"
                x_min = int(rect[0] * vid_width + 0.1 * vid_width)
                y_min = int(rect[2] * vid_height + 0.1 * vid_height)
                x_max = int(rect[1] * vid_width + 0.1 * vid_width)
                y_max = int(rect[3] * vid_height + 0.1 * vid_height)
            else:
                x_min = int(rect[0] * vid_width)
                y_min = int(rect[2] * vid_height)
                x_max = int(rect[1] * vid_width)
                y_max = int(rect[3] * vid_height)
            y_mid = (y_min + y_max) / 2.
            x_mid = (x_min + x_max) / 2.
            len_ = max(x_max - x_min, y_max - y_min)
            face_rect = [x_mid - len_, y_mid - len_, x_mid + len_, y_mid + len_]

        x_min, y_min, x_max, y_max = face_rect
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        crop_size = int(max(seq_w * 1.35, seq_h * 1.35))
        x_min = int(max(0, x_mid - crop_size * 0.5))
        y_min = int(max(0, y_mid - crop_size * 0.45))
        x_max = int(min(vid_width, x_min + crop_size))
        y_max = int(min(vid_height, y_min + crop_size))

        frame_face = frame[y_min:y_max, x_min:x_max]
        # print(y_min, y_max, x_min, x_max)
        # cv2.imshow("s", frame_face)
        # cv2.waitKey(10)
        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])

        # point_size = 1
        # point_color = (0, 0, 255)  # BGR
        # thickness = 4  # 0 、4、8
        # for coor in pts_3d[frame_index]:
        #     # coor = (coor +1 )/2.
        #     cv2.circle(frame, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
        # cv2.imshow("a", frame)
        # cv2.waitKey(30)
    cap.release()  # 释放视频对象
    return pts_3d


def PrepareVideo(video_in_path, video_out_path, face_rect=[200, 200, 520, 520], resize_option = False):
    # 1 视频转换为25FPS
    if resize_option:
        cap = cv2.VideoCapture(video_in_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(720 / width, 1280 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # 确保新的宽高为偶数
        new_width = new_width //2*2
        new_height = new_height //2*2
        cap.release()
        # 补全 FFmpeg 命令
        ffmpeg_cmd = "ffmpeg -i {} -vf \"scale={}:{}\" -r 25 -an -y {}".format(video_in_path, new_width, new_height,
                                                                               video_out_path)
    else:
        ffmpeg_cmd = "ffmpeg -i {} -r 25 -an -y {}".format(video_in_path, video_out_path)
    os.system(ffmpeg_cmd)
    print(ffmpeg_cmd)

    cap = cv2.VideoCapture(video_out_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    print("视频帧数：", frames)
    pts_3d = ExtractFromVideo(video_out_path, face_rect)
    assert type(pts_3d) is np.ndarray and len(pts_3d) == frames,"关键点已提取"
    Path_output_pkl = video_out_path[:-4] + ".pkl"
    with open(Path_output_pkl, "wb") as f:
        pickle.dump(pts_3d, f)

def data_preparation_mini(video, video_dir_path, resize_option = False):
    new_data_path = os.path.join(video_dir_path, "data")
    os.makedirs(new_data_path, exist_ok=True)
    video_out_path = "{}/circle.mp4".format(new_data_path)
    # CirculateVideo(video_mouthClose, video_out_path, face_rect=[290, 190, 440, 350])
    PrepareVideo(video, video_out_path, face_rect=None, resize_option = resize_option)

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 3:
        print("Usage: python data_preparation_mini.py <静默视频> <输出文件夹位置>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video = sys.argv[1]
    video_dir_path = sys.argv[2]
    print(f"Video dir path is set to: {video_dir_path}")
    data_preparation_mini(video, video_dir_path)


if __name__ == "__main__":
    main()
