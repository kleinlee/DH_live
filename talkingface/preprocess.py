import shutil
import numpy as np
import cv2
import os
import sys
import time
import argparse
from talkingface.run_utils import video_pts_process, concat_output_2binfile
from talkingface.mediapipe_utils import detect_face_mesh,detect_face
from talkingface.utils import main_keypoints_index,INDEX_LIPS
# 1、是否是mp4，宽高是否大于200，时长是否大于2s,可否成功转换为符合格式的mp4
# 2、面部关键点检测及是否可以构成循环视频
# 4、旋转矩阵、面部mask估计
# 5、验证文件完整性

dir_ = "data/asset/Actor"
def print_log(task_id, progress, status, Error, mode = 0):
    '''
    status: -1代表未开始， 0代表处理中， 1代表已完成， 2代表出错中断
    progress： 0-1000， 进度千分比
    '''
    print("task_id: {}. progress: {:0>4d}. status: {}. mode: {}. Error: {}".format(task_id, progress, status, mode, Error))
    sys.stdout.flush()

def check_step0(task_id, video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if vid_width < 200 or vid_height < 200:
            print_log(task_id, 0, 2, "video width/height < 200")
            return 0
        if frames < 2*fps:
            print_log(task_id, 0, 2, "video duration < 2s")
            return 0
        os.makedirs(os.path.join(dir_, task_id), exist_ok=True)
        front_video_path = os.path.join("data", "front.mp4")
        scale = max(vid_width / 720., vid_height / 1280.)
        if scale > 1:
            new_width = int(vid_width / scale + 0.1)//2 * 2
            new_height = int(vid_height / scale + 0.1)//2 * 2
            ffmpeg_cmd = "ffmpeg -i {} -r 25 -ss 00:00:00 -t 00:02:00 -vf scale={}:{} -an -loglevel quiet -y {}".format(
                video_path,new_width,new_height,front_video_path)
        else:
            ffmpeg_cmd = "ffmpeg -i {} -r 25 -ss 00:00:00 -t 00:02:00 -an -loglevel quiet -y {}".format(
                video_path, front_video_path)
        os.system(ffmpeg_cmd)
        if not os.path.isfile(front_video_path):
            return 0
        return 1
    except:
        print_log(task_id, 0, 2, "video cant be opened")
        return 0

def check_step1(task_id):
    front_video_path = os.path.join("data", "front.mp4")
    back_video_path = os.path.join("data", "back.mp4")
    video_out_path = os.path.join(dir_, task_id, "video.mp4")
    face_info_path = os.path.join(dir_, task_id, "video_info.bin")
    preview_path = os.path.join(dir_, task_id, "preview.jpg")
    if ExtractFromVideo(task_id, front_video_path) != 1:
        shutil.rmtree(os.path.join(dir_, task_id))
        return 0
    ffmpeg_cmd = "ffmpeg -i {} -vf reverse -loglevel quiet -y {}".format(front_video_path, back_video_path)
    os.system(ffmpeg_cmd)
    ffmpeg_cmd = "ffmpeg -f concat -i {} -loglevel quiet -y {}".format("data/video_concat.txt", video_out_path)
    os.system(ffmpeg_cmd)
    ffmpeg_cmd = "ffmpeg -i {} -vf crop=w='min(iw\,ih)':h='min(iw\,ih)',scale=256:256,setsar=1 -vframes 1 {}".format(front_video_path, preview_path)
    # ffmpeg_cmd = "ffmpeg -i {} -vf scale=256:-1 -loglevel quiet -y {}".format(front_video_path, preview_path)
    os.system(ffmpeg_cmd)
    if os.path.isfile(front_video_path):
        os.remove(front_video_path)
    if os.path.isfile(back_video_path):
        os.remove(back_video_path)
    if os.path.isfile(video_out_path) and os.path.isfile(face_info_path):
        return 1
    else:
        return 0

# def check_step2(task_id, ):
#     mat_list, pts_normalized_list, face_mask_pts = video_pts_process(pts_array_origin)


def ExtractFromVideo(task_id, front_video_path):
    cap = cv2.VideoCapture(front_video_path)
    if not cap.isOpened():
        print_log(task_id, 0, 2, "front_video cant be opened by opencv")
        return -1

    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    totalFrames = int(totalFrames)
    pts_3d = np.zeros([totalFrames, 478, 3])
    frame_index = 0
    face_rect_list = []
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()  # 按帧读取视频
        # #到视频结尾时终止
        if ret is False:
            break
        rect_2d = detect_face([frame])
        rect = rect_2d[0]
        tag_ = 1 if np.sum(rect) > 0 else 0
        if frame_index == 0 and tag_ != 1:
            print_log(task_id, 0, 2, "no face detected in first frame")
            cap.release()  # 释放视频对象
            return 0
        elif tag_ == 0:  # 有时候人脸检测会失败，就用上一帧的结果替代这一帧的结果
            rect = face_rect_list[-1]

        face_rect_list.append(rect)

        x_min = rect[0] * vid_width
        y_min = rect[2] * vid_height
        x_max = rect[1] * vid_width
        y_max = rect[3] * vid_height
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        x_min = int(max(0, x_mid - seq_w * 0.65))
        y_min = int(max(0, y_mid - seq_h * 0.4))
        x_max = int(min(vid_width, x_mid + seq_w * 0.65))
        y_max = int(min(vid_height, y_mid + seq_h * 0.8))

        frame_face = frame[y_min:y_max, x_min:x_max]
        frame_kps = detect_face_mesh([frame_face])[0]
        if np.sum(frame_kps) == 0:
            print_log(task_id, 0, 2, "Frame num {} keypoint error".format(frame_index))
            cap.release()  # 释放视频对象
            return 0
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])
        frame_index += 1

        if time.time() - start_time > 0.5:
            progress = int(1000 * frame_index / totalFrames * 0.99)
            print_log(task_id, progress, 0, "handling...")
            start_time = time.time()
    cap.release()  # 释放视频对象
    if type(pts_3d) is np.ndarray and len(pts_3d) == totalFrames:
        pts_3d_main = pts_3d[:, main_keypoints_index]
        mat_list, pts_normalized_list, face_pts_mean_personal, face_mask_pts_normalized = video_pts_process(pts_3d_main)

        output = concat_output_2binfile(mat_list, pts_3d, face_pts_mean_personal, face_mask_pts_normalized)
        # print(output.shape)
        pts_normalized_list = np.array(pts_normalized_list)[:, INDEX_LIPS]
        # 找出此模特正面人脸的嘴巴区域范围
        x_max, x_min = np.max(pts_normalized_list[:, :, 0]), np.min(pts_normalized_list[:, :, 0])
        y_max, y_min = np.max(pts_normalized_list[:, :, 1]), np.min(pts_normalized_list[:, :, 1])
        y_min = y_min + (y_max - y_min) / 10.

        first_line = np.zeros([406])
        first_line[:4] = np.array([x_min,x_max,y_min,y_max])
        # print(first_line)
        #
        # pts_2d_main = pts_3d[:, main_keypoints_index, :2].reshape(len(pts_3d), -1)
        # smooth_array_ = np.array(mat_list).reshape(-1, 16)*100
        #
        # output = np.concatenate([smooth_array_, pts_2d_main], axis=1).astype(np.float32)
        output = np.concatenate([first_line.reshape(1,-1), output], axis=0).astype(np.float32)
        # print(smooth_array_.shape, pts_2d_main.shape, first_line.shape, output.shape)
        face_info_path = os.path.join(dir_, task_id, "video_info.bin")
        # np.savetxt(face_info_path, output, fmt='%.1f')
        # print(222)
        output.tofile(face_info_path)
        return 1
    else:
        print_log(task_id, 0, 2, "keypoint cant be saved")
        return 0

def check_step0_audio(task_id, video_path):
    dir_ = "data/asset/Audio"
    wav_path = os.path.join(dir_, task_id + ".wav")
    ffmpeg_cmd = "ffmpeg -i {} -ac 1 -ar 16000 -loglevel quiet -y {}".format(
                video_path, wav_path)
    os.system(ffmpeg_cmd)
    if not os.path.isfile(wav_path):
        print_log(task_id, 0, 2, "audio convert failed", 2)
        return 0
    return 1

def new_task(task_id, task_mode, video_path):
    # print(task_id, task_mode, video_path)
    if task_mode == "0":  # "actor"
        print_log(task_id, 0, 0, "handling...")
        if check_step0(task_id, video_path):
            print_log(task_id, 0, 0, "handling...")
            if check_step1(task_id):
                print_log(task_id, 1000, 1, "process finished, click to confirm")
    if task_mode == "2":   # "audio"
        print_log(task_id, 0, 0, "handling...", 2)
        if check_step0_audio(task_id, video_path):
            print_log(task_id, 1000, 1, "process finished, click to confirm", 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code to preprocess videos')
    parser.add_argument('--task_id', type=str, help='task_id')
    parser.add_argument('--task_mode', type=str, help='task_mode')
    parser.add_argument('--video_path', type=str, help='Filepath of video that contains faces to use')
    args = parser.parse_args()
    new_task(args.task_id, args.task_mode, args.video_path)