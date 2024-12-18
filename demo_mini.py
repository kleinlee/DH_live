import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import os
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap
current_dir = os.path.dirname(os.path.abspath(__file__))
from mini_live.render import create_render_model
import pickle
import cv2
import time
import numpy as np
import glob
import random
import os
import sys
import torch
from talkingface.model_utils import LoadAudioModel, Audio2bs
from talkingface.data.few_shot_dataset import get_image

def run(video_path, pkl_path, wav_path, output_video_path):
    Audio2FeatureModel = LoadAudioModel(r'checkpoint/lstm/lstm_model_epoch_325.pkl')

    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("checkpoint/DINet_mini/epoch_40.pth")

    standard_size = 256
    crop_rotio = [0.5, 0.5, 0.5, 0.5]
    out_w = int(standard_size * (crop_rotio[0] + crop_rotio[1]))
    out_h = int(standard_size * (crop_rotio[2] + crop_rotio[3]))
    out_size = (out_w, out_h)
    renderModel_gl = create_render_model((out_w, out_h), floor=20)

    from mini_live.obj.obj_utils import generateWrapModel
    from talkingface.utils import crop_mouth, main_keypoints_index
    wrapModel, wrapModel_face = generateWrapModel()

    with open(pkl_path, "rb") as f:
        images_info = pickle.load(f)

    images_info = np.concatenate([images_info, images_info[::-1]], axis=0)

    cap = cv2.VideoCapture(video_path)
    vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    list_source_crop_rect = []
    list_video_img = []
    list_standard_img = []
    list_standard_v = []
    list_standard_vt = []
    for frame_index in range(min(vid_frame_count, len(images_info))):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        source_pts = images_info[frame_index]
        source_crop_rect = crop_mouth(source_pts[main_keypoints_index], vid_width, vid_height)

        standard_img = get_image(frame, source_crop_rect, input_type="image", resize=standard_size)
        standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize=standard_size)
        standard_vt = standard_v[:, :2] / standard_size

        list_video_img.append(frame)
        list_source_crop_rect.append(source_crop_rect)
        list_standard_img.append(standard_img)
        list_standard_v.append(standard_v)
        list_standard_vt.append(standard_vt)
    cap.release()

    renderModel_mini.reset_charactor(list_standard_img, np.array(list_standard_v)[:, main_keypoints_index])
    from talkingface.run_utils import calc_face_mat
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v), renderModel_gl.face_pts_mean)
    from mini_live.obj.wrap_utils import newWrapModel
    face_wrap_entity = newWrapModel(wrapModel, face_pts_mean_personal_primer)

    renderModel_gl.GenVBO(face_wrap_entity)

    bs_array = Audio2bs(wav_path, Audio2FeatureModel)[5:] * 0.5
    import uuid
    task_id = str(uuid.uuid1())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "{}.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))

    for frame_index in range(len(mat_list)):
        if frame_index >= len(bs_array):
            continue
        bs = np.zeros([12], dtype=np.float32)
        bs[:6] = bs_array[frame_index, :6]
        # bs[2] = frame_index* 5

        verts_frame_buffer = np.array(list_standard_vt)[frame_index, index_wrap, :2].copy() * 2 - 1

        rgba = renderModel_gl.render2cv(verts_frame_buffer, out_size=out_size, mat_world=mat_list[frame_index].T,
                                        bs_array=bs)
        # rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        # rgba = cv2.resize(rgba, (128, 128))
        rgba = rgba[::2, ::2, :]

        gl_tensor = torch.from_numpy(rgba / 255.).float().permute(2, 0, 1).unsqueeze(0)
        source_tensor = cv2.resize(list_standard_img[frame_index], (128, 128))
        source_tensor = torch.from_numpy(source_tensor / 255.).float().permute(2, 0, 1).unsqueeze(0)

        warped_img = renderModel_mini.interface(source_tensor.cuda(), gl_tensor.cuda())

        image_numpy = warped_img.detach().squeeze(0).cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)

        x_min, y_min, x_max, y_max = list_source_crop_rect[frame_index]

        img_face = cv2.resize(image_numpy, (x_max - x_min, y_max - y_min))
        img_bg = list_video_img[frame_index][:, :, :3]
        img_bg[y_min:y_max, x_min:x_max, :3] = img_face[:, :, :3]
        # cv2.imshow('scene', img_bg[:,:,::-1])
        # cv2.waitKey(10)
        # print(time.time())

        videoWriter.write(img_bg[:, :, ::-1])
    videoWriter.release()

    os.system(
        "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p {}".format(save_path, wav_path, output_video_path))
    os.remove(save_path)

    cv2.destroyAllWindows()

def main():
    # 检查命令行参数的数量
    if len(sys.argv) < 4:
        print("Usage: python demo_mini.py <video_path> <audio_path> <output_video_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_path = sys.argv[1]
    print(f"Video path is set to: {video_path}")
    wav_path = sys.argv[2]
    print(f"Audio path is set to: {wav_path}")
    output_video_name = sys.argv[3]
    print(f"output video name is set to: {output_video_name}")

    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)

    run(video_path, pkl_path, wav_path, output_video_name)

# 示例使用
if __name__ == "__main__":
    main()



