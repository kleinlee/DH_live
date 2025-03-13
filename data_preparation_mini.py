import subprocess
import tqdm
import numpy as np
import cv2
import sys
import os
import math
import pickle
import mediapipe as mp
import shutil

# 自定义异常类
class VideoProcessingError(Exception):
    """视频处理基类异常"""
    pass

class FFmpegError(VideoProcessingError):
    """FFmpeg处理异常"""
    pass

class FaceDetectionError(VideoProcessingError):
    """人脸检测异常"""
    pass

class FirstFrameFaceDetectionError(FaceDetectionError):
    """首帧人脸检测异常"""
    pass

class FaceMeshDetectionError(VideoProcessingError):
    """面部网格检测异常"""
    pass

class EnvironmentError(VideoProcessingError):
    """环境配置错误"""
    pass

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


def detect_face(frame: np.ndarray, min_detection_confidence: float = 0.5) -> list:
    """人脸检测并验证有效性"""
    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
    ) as face_detection:

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 人脸数量检查
        if not results.detections:
            raise FaceDetectionError("未检测到人脸")
        if len(results.detections) > 1:
            raise FaceDetectionError("检测到多个人脸")

        detection = results.detections[0]
        rect = detection.location_data.relative_bounding_box
        out_rect = [
            rect.xmin,
            rect.xmin + rect.width,
            rect.ymin,
            rect.ymin + rect.height
        ]

        # 关键点验证
        nose = mp_face_detection.get_key_point(
            detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
        left_eye = mp_face_detection.get_key_point(
            detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
        right_eye = mp_face_detection.get_key_point(
            detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

        if nose.x > left_eye.x or nose.x < right_eye.x:
            raise FaceDetectionError("人脸角度不符合要求，请提供正脸图片")

        # 边界检查
        h, w = frame.shape[:2]
        if (out_rect[0] < 0 or out_rect[2] < 0
                or out_rect[1] > 1 or out_rect[3] > 1):
            raise FaceDetectionError("人脸区域超出画面边界")

        # 尺寸检查
        if rect.width * w < 80 or rect.height * h < 80:
            raise FaceDetectionError("人脸尺寸不能低于80*80像素")

        return out_rect


def calc_face_interact(face0, face1):
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / ((x_max - x_min) * (y_max - y_min))
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / ((x_max - x_min) * (y_max - y_min))
    return min(tmp0, tmp1)


def detect_face_mesh(frame: np.ndarray) -> np.ndarray:
    """面部网格检测"""
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts_3d = np.zeros((478, 3))

        if not results.multi_face_landmarks:
            raise FaceMeshDetectionError("未检测到面部网格")

        image_height, image_width = frame.shape[:2]
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            pts_3d[idx] = [
                min(math.floor(landmark.x * image_width), image_width - 1),
                min(math.floor(landmark.y * image_height), image_height - 1),
                min(math.floor(landmark.z * image_width), image_width - 1)
            ]
        return pts_3d


def extract_from_video(
        video_path: str,
        output_pkl_path: str
) -> None:
    """从视频提取关键点"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoProcessingError("无法打开视频文件")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        pts_3d = np.zeros((total_frames, 478, 3))
        face_rect = None
        for frame_index in tqdm.tqdm(range(total_frames)):
            ret, frame = cap.read()  # 按帧读取视频
            # #到视频结尾时终止
            if ret is False:
                break

            if frame_index == 0:
                try:
                    rect = detect_face(frame, 0.25)
                    x_min = int(rect[0] * vid_width)
                    y_min = int(rect[2] * vid_height)
                    x_max = int(rect[1] * vid_width)
                    y_max = int(rect[3] * vid_height)
                except FaceDetectionError:
                    # 尝试裁剪后检测
                    cropped = frame[
                              int(0.1 * vid_height):int(0.9 * vid_height),
                              int(0.1 * vid_width):int(0.9 * vid_width)
                              ]
                    try:
                        rect = detect_face(cropped, 0.25)
                    except FaceDetectionError as e:
                        raise FirstFrameFaceDetectionError("首帧人脸检测失败") from e

                    # 转换坐标到原图
                    x_min = int(rect[0] * vid_width + 0.1 * vid_width)
                    y_min = int(rect[2] * vid_height + 0.1 * vid_height)
                    x_max = int(rect[1] * vid_width + 0.1 * vid_width)
                    y_max = int(rect[3] * vid_height + 0.1 * vid_height)

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
                face_rect = (x_min, y_min, x_max, y_max)

            # 裁剪人脸区域
            x0, y0, x1, y1 = face_rect
            face_region = frame[y0:y1, x0:x1]
            # print(y_min, y_max, x_min, x_max)
            # cv2.imshow("s", frame_face)
            # cv2.waitKey(10)
            try:
                frame_kps = detect_face_mesh(face_region)
            except FaceMeshDetectionError as e:
                raise VideoProcessingError(f"第{frame_index}帧面部网格检测失败") from e
            pts_3d[frame_index] = frame_kps + [x0, y0, 0]



            # point_size = 1
            # point_color = (0, 0, 255)  # BGR
            # thickness = 4  # 0 、4、8
            # for coor in pts_3d[frame_index]:
            #     # coor = (coor +1 )/2.
            #     cv2.circle(frame, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
            # cv2.imshow("a", frame)
            # cv2.waitKey(30)
        # 保存关键点
        with open(output_pkl_path, "wb") as f:
            pickle.dump(pts_3d, f)
    finally:
        cap.release()  # 释放视频对象
    return pts_3d


def prepare_video(
        input_path: str,
        output_path: str,
        resize_option: bool = False
) -> int:
    # 1 视频转换为25FPS
    if resize_option:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(720 / width, 1280 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # 确保新的宽高为偶数
        new_width = new_width //2*2
        new_height = new_height //2*2
        cap.release()
        vf_arg = f"scale={new_width}:{new_height}"
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", vf_arg,
            "-r", "25", "-an", "-y", output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-r", "25", "-an", "-y", output_path
        ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True
        )
        return 0
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"FFmpeg处理失败: {e.stderr}") from e



def data_preparation_mini(input_video, video_dir_path, resize_option = False):
    # 检测系统环境是否有ffmpeg
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg未安装或不在PATH中，请安装ffmpeg并设置为环境变量")

    # 创建输出目录
    data_dir = os.path.join(video_dir_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 预处理视频
    output_video = os.path.join(data_dir, "processed.mp4")
    prepare_video(input_video, output_video, resize_option = resize_option)

    # 提取关键点
    output_pkl = output_video.replace(".mp4", ".pkl")
    extract_from_video(output_video, output_pkl)
    result = {
        "status": "success",
        "output_video": output_video,
        "output_pkl": output_pkl
    }
    return result

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
    print("Done!")


if __name__ == "__main__":
    main()
