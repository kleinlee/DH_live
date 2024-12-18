import uuid
import tqdm
import numpy as np
import cv2
import sys
import os
import math
from talkingface.data.few_shot_dataset import get_image
import torch
import mediapipe as mp
from talkingface.utils import crop_mouth, main_keypoints_index, smooth_array
import json
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

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


def ExtractFromVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    dir_path = os.path.dirname(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    totalFrames = int(totalFrames)
    pts_3d = np.zeros([totalFrames, 478, 3])
    frame_index = 0
    face_rect_list = []
    ms_list = []
    model_name = os.path.basename(video_path)[:-4]

    # os.makedirs("../preparation/{}/image".format(model_name))
    for frame_index in tqdm.tqdm(range(totalFrames)):
        ret, frame = cap.read()  # 按帧读取视频
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ms_list.append(ms)
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
        print(y_min, y_max, x_min, x_max)
        # cv2.imshow("s", frame_face)
        # cv2.waitKey(20)
        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])
    cap.release()  # 释放视频对象
    return pts_3d,ms_list


def step0_keypoints(video_in_path, out_path):
    video_out_path = os.path.join(out_path, "01.mp4")
    # 1 视频转换为25FPS
    ffmpeg_cmd = "ffmpeg -i {} -r 25 -an -y {}".format(video_in_path, video_out_path)
    os.system(ffmpeg_cmd)

    # front_video_path = video_in_path

    cap = cv2.VideoCapture(video_out_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    print("正向视频帧数：", frames)
    pts_3d,_ = ExtractFromVideo(video_out_path)
    if type(pts_3d) is np.ndarray and len(pts_3d) == frames:
        print("关键点已提取")

    pts_3d = pts_3d.reshape(len(pts_3d), -1)
    smooth_array_ = smooth_array(pts_3d, weight=[0.03, 0.06, 0.11, 0.6, 0.11, 0.06, 0.03])
    pts_3d = smooth_array_.reshape(len(pts_3d), 478, 3)
    return pts_3d,vid_width,vid_height

def step1_crop_mouth(pts_3d, vid_width, vid_height):
    list_source_crop_rect = [crop_mouth(source_pts[main_keypoints_index], vid_width, vid_height) for source_pts in
                             pts_3d]
    list_source_crop_rect = np.array(list_source_crop_rect).reshape(len(pts_3d), -1)
    face_size = (list_source_crop_rect[:,2] - list_source_crop_rect[:,0]).mean()/2.0 + (list_source_crop_rect[:,3] - list_source_crop_rect[:,1]).mean()/2.0
    face_size = int(face_size)//2 * 2
    face_mid = (list_source_crop_rect[:,2:] + list_source_crop_rect[:,0:2])/2.
    # step 1: Smooth Cropping Rectangle Transition
    # Since HTML video playback can have inconsistent frame rates and may not align precisely from frame to frame, adjust the cropping rectangle to transition smoothly, compensating for potential misalignment.
    face_mid = smooth_array(face_mid, weight=[0.20, 0.20, 0.20, 0.20, 0.20])
    face_mid = face_mid.astype(int)
    if face_mid[:, 0].max() + face_size / 2 > vid_width or face_mid[:, 1].max() + face_size / 2 > vid_height:
        raise ValueError("人脸范围超出了视频，请保证视频合格后再重试")

    list_source_crop_rect = np.concatenate([face_mid - face_size // 2, face_mid + face_size // 2], axis = 1)

    standard_size = 128
    list_standard_v = []
    for frame_index in range(len(list_source_crop_rect)):
        source_pts = pts_3d[frame_index]
        source_crop_rect = list_source_crop_rect[frame_index]
        print(source_crop_rect)
        standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize=standard_size)

        list_standard_v.append(standard_v)

    return list_source_crop_rect, list_standard_v

def step2_generate_obj(list_source_crop_rect, list_standard_v, out_path):
    from mini_live.obj.obj_utils import generateRenderInfo, generateWrapModel
    render_verts, render_face = generateRenderInfo()
    face_pts_mean = render_verts[:478, :3].copy()

    wrapModel_verts, wrapModel_face = generateWrapModel()
    # 求平均人脸
    from talkingface.run_utils import calc_face_mat
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v), face_pts_mean)
    from mini_live.obj.wrap_utils import newWrapModel

    face_wrap_entity = newWrapModel(wrapModel_verts, face_pts_mean_personal_primer)

    with open(os.path.join(out_path,"face3D.obj"), "w") as f:
        for i in face_wrap_entity:
            f.write("v {:.3f} {:.3f} {:.3f} {:.02f} {:.0f}\n".format(i[0], i[1], i[2], i[3], i[4]))
        for i in range(len(wrapModel_face) // 3):
            f.write("f {0} {1} {2}\n".format(wrapModel_face[3 * i] + 1, wrapModel_face[3 * i + 1] + 1,
                                             wrapModel_face[3 * i + 2] + 1))
    json_data = []
    for frame_index in range(len(list_source_crop_rect)):
        source_crop_rect = list_source_crop_rect[frame_index]
        standard_v = list_standard_v[frame_index]

        standard_v = standard_v[index_wrap, :2].flatten().tolist()
        mat = mat_list[frame_index].T.flatten().tolist()
        # 将 standard_v 中所有元素四舍五入到两位小数
        standard_v_rounded = [round(i, 5) for i in mat] + [round(i, 1) for i in standard_v]
        print(len(standard_v_rounded), 16 + 209 * 2)
        json_data.append({"rect": source_crop_rect.tolist(), "points": standard_v_rounded})
        # print(json_data)
        # break
    with open(os.path.join(out_path, "json_data.json"), "w") as f:
        json.dump(json_data, f)

def step3_generate_ref_tensor(list_source_crop_rect, list_standard_v, out_path):
    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("checkpoint/DINet_mini/epoch_40.pth")

    from talkingface.data.few_shot_dataset import select_ref_index
    from talkingface.utils import draw_mouth_maps

    driven_keypoints = np.array(list_standard_v)[:,main_keypoints_index]
    ref_img_index_list = select_ref_index(driven_keypoints, n_ref=3, ratio=0.33)  # 从当前视频选n_ref个图片
    ref_img_list = []

    video_path = os.path.join(out_path, "01.mp4")
    cap_input = cv2.VideoCapture(video_path)
    for index in ref_img_index_list:
        cap_input.set(cv2.CAP_PROP_POS_FRAMES, index)  # 设置要获取的帧号
        ret, frame = cap_input.read()

        ref_face_edge = draw_mouth_maps(driven_keypoints[index], size=(128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ref_img = get_image(frame, list_source_crop_rect[index], input_type="image", resize=128)

        ref_face_edge = cv2.resize(ref_face_edge, (128, 128))
        ref_img = cv2.resize(ref_img, (128, 128))
        w_pad = int((128 - 72) / 2)
        h_pad = int((128 - 56) / 2)

        ref_img = np.concatenate([ref_img[h_pad:-h_pad, w_pad:-w_pad], ref_face_edge[h_pad:-h_pad, w_pad:-w_pad, :1]],
                                 axis=2)
        # cv2.imshow("ss", ref_img[:,:,::-1])
        # cv2.waitKey(-1)
        ref_img_list.append(ref_img)
    ref_img = np.concatenate(ref_img_list, axis=2)

    ref_tensor = torch.from_numpy(ref_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()

    renderModel_mini.net.ref_input(ref_tensor)

    ref_in_feature = renderModel_mini.net.infer_model.ref_in_feature
    print(1111, ref_in_feature.size())
    ref_in_feature = ref_in_feature.detach().squeeze(0).cpu().float().numpy().flatten()
    print(1111, ref_in_feature.shape)

    np.savetxt(os.path.join(out_path, 'ref_data.txt'), ref_in_feature, fmt='%.8f')

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 2:
        print("Usage: python data_preparation_web.py <video_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_in_path = sys.argv[1]

    # video_in_path = r"E:\data\video\video/5.mp4"
    out_path = "web_demo/static/assets"
    pts_3d, vid_width,vid_height = step0_keypoints(video_in_path, out_path)
    list_source_crop_rect, list_standard_v = step1_crop_mouth(pts_3d, vid_width, vid_height)
    step2_generate_obj(list_source_crop_rect, list_standard_v, out_path)
    step3_generate_ref_tensor(list_source_crop_rect, list_standard_v, out_path)

if __name__ == "__main__":
    main()
