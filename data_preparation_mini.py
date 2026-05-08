import subprocess
import tqdm
import numpy as np
import cv2
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽 INFO 和 WARNING
os.environ['GLOG_minloglevel'] = '2'      # 屏蔽 glog 日志
import math
import pickle
import mediapipe as mp
import shutil
import glob
MODULO_N = 16
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

def encode_binary_pixels(frame, width, modulo_value):
    """
    在右上角2x2区域编码二进制序号

    2x2编码区域（右上角）:
    [y, x-1]   [y, x]     <- 位1, 位0
    [y+1, x-1] [y+1, x]   <- 位3, 位2

    缓冲区域与相邻像素一致
    """
    for bit in range(4):
        is_white = (modulo_value >> bit) & 1
        color = 255 if is_white else 0

        if bit == 0:
            dy, dx = 0, 0
        elif bit == 1:
            dy, dx = 0, 1
        elif bit == 2:
            dy, dx = 1, 0
        elif bit == 3:
            dy, dx = 1, 1
        else:
            dy, dx = 0, 0

        frame[dy, width - 1 - dx] = [color, color, color]

def extract_from_video(
        data_dir: str,
        output_pkl_path: str,
        output_video_path: str,
        matting: bool,
        reverse_option: bool
) -> None:
    """从视频提取关键点"""
    img_list = glob.glob(os.path.join(data_dir, "*.png"))
    img_list.sort()   # 按序号排序
    vid_width = 0
    vid_height = 0
    if 1:
        total_frames = len(img_list)
        pts_3d = np.zeros((total_frames, 478, 3))
        face_rect = None
        for frame_index in tqdm.tqdm(range(total_frames)):
            frame_bgr = cv2.imread(img_list[frame_index])
            vid_width = frame_bgr.shape[1]
            vid_height = frame_bgr.shape[0]
            if frame_index == 0:
                try:
                    rect = detect_face(frame_bgr[:, :, :3], 0.25)
                    x_min = int(rect[0] * vid_width)
                    y_min = int(rect[2] * vid_height)
                    x_max = int(rect[1] * vid_width)
                    y_max = int(rect[3] * vid_height)
                except FaceDetectionError:
                    # 尝试裁剪后检测
                    cropped = frame_bgr[
                              int(0.1 * vid_height):int(0.9 * vid_height),
                              int(0.1 * vid_width):int(0.9 * vid_width),
                              :3]
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
                crop_size = max(x_max - x_min, y_max - y_min) * 0.8
                x_min = int(max(0, x_mid - crop_size))
                y_min = int(max(0, y_mid - crop_size))
                x_max = int(min(vid_width, x_mid + crop_size))
                y_max = int(min(vid_height, y_mid + crop_size))
                face_rect = (x_min, y_min, x_max, y_max)

            # 裁剪人脸区域
            x0, y0, x1, y1 = face_rect
            face_region = frame_bgr[y0:y1, x0:x1, :3]
            # print(y_min, y_max, x_min, x_max)
            # cv2.imshow("s", frame_face)
            # cv2.waitKey(10)
            try:
                frame_kps = detect_face_mesh(face_region)
            except FaceMeshDetectionError as e:
                raise VideoProcessingError(f"第{frame_index}帧面部网格检测失败") from e
            pts_3d[frame_index] = frame_kps + [x0, y0, 0]

            # 根据frame_kps更新face_rect
            x_min, y_min, x_max, y_max = frame_kps[:, 0].min(), frame_kps[:, 1].min(), frame_kps[:, 0].max(), frame_kps[:, 1].max()
            x_min, y_min, x_max, y_max = x0+x_min, y0+y_min, x0+x_max, y0+y_max
            x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
            crop_size = max(x_max - x_min, y_max - y_min) * 0.8
            x_min = int(max(0, x_mid - crop_size))
            y_min = int(max(0, y_mid - crop_size))
            x_max = int(min(vid_width, x_mid + crop_size))
            y_max = int(min(vid_height, y_mid + crop_size))
            face_rect = (x_min, y_min, x_max, y_max)
            if frame_index > 0:
                # 2. 计算相邻帧之间 XY 坐标的移动距离，超出一定范围就认定不合理
                frame_diff = pts_3d[frame_index] - pts_3d[frame_index - 1]
                xy_displacement = np.sqrt(frame_diff[:, 0] ** 2 + frame_diff[:, 1] ** 2)
                xy_displacement = xy_displacement.mean()
                if xy_displacement > crop_size/6:
                    cv2.imshow("frame", frame_bgr)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    raise VideoProcessingError(f"第{frame_index}帧面部范围大幅度改变，请检查")


            if matting:
                from talkingface.RVM import process_img_matting
                final_rgba = process_img_matting(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA), frame_index == 0)
                green_bgr = np.zeros((final_rgba.shape[0], final_rgba.shape[1], 3), dtype=np.uint8)
                green_bgr[:, :, 1] = 255

                alpha = final_rgba[:, :, 3:4] / 255.0  # Normalize alpha to [0, 1]
                final_bgr = (green_bgr * (1 - alpha) + final_rgba[:, :, :3][:, :, ::-1] * alpha).astype(np.uint8)
            else:
                final_bgr = frame_bgr

            modulo_value = frame_index % MODULO_N
            encode_binary_pixels(final_bgr, vid_width, modulo_value)

            cv2.imwrite(os.path.join(data_dir, f"{frame_index:06d}.png"), final_bgr)

            if reverse_option:
                frame_count_inverse = total_frames * 2 - frame_index - 1
                modulo_value = frame_count_inverse % MODULO_N
                encode_binary_pixels(final_bgr, vid_width, modulo_value)
                cv2.imwrite(os.path.join(data_dir, f"{frame_count_inverse:06d}.png"), final_bgr)

        # 保存关键点
        with open(output_pkl_path, "wb") as f:
            pickle.dump(pts_3d, f)

        fps = 25
        crf = 18
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(data_dir, '%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
    return total_frames


def prepare_video(
        input_path: str,
        output_path: str,
        resize_option: bool = False
) -> int:
    if resize_option:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        print(f"video info: width-{width} height-{height} rotate_code-{rotate_code}")
        if rotate_code == 90 or rotate_code == 270:
            width, height = height, width
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
            "-r", "25", '-f', 'image2', "-y", os.path.join(output_path, '%06d.png')
        ]
    else:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-r", "25", '-f', 'image2', "-y", os.path.join(output_path, '%06d.png')
        ]

    print("ffmpeg cmd: ", cmd)
    # Run the command
    subprocess.run(cmd, check=True)

    # Count the number of frames generated
    frame_count = len([f for f in os.listdir(output_path) if f.endswith('.png')])
    return frame_count



def data_preparation_mini(input_video, video_dir_path,  matting = False, resize_option = False, reverse_option = True):
    # 检测系统环境是否有ffmpeg
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg未安装或不在PATH中，请安装ffmpeg并设置为环境变量")

    # 创建输出目录
    data_dir = os.path.join(video_dir_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    frames_png_dir = os.path.join(video_dir_path, "frames")
    os.makedirs(frames_png_dir, exist_ok=True)

    frame_count = prepare_video(input_video, frames_png_dir, resize_option = resize_option)

    # 提取关键点
    output_pkl_path = os.path.join(data_dir, "processed.pkl")
    output_video_path = os.path.join(data_dir, "processed.mp4")
    extract_from_video(frames_png_dir, output_pkl_path, output_video_path, matting, reverse_option)
    shutil.rmtree(frames_png_dir)
    result = {
        "status": "success",
        "frame_count": frame_count,
        "output_video": output_video_path
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
    data_preparation_mini(video, video_dir_path, matting = True)
    print("Done!")


if __name__ == "__main__":
    main()
