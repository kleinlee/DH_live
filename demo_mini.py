import os
import uuid
import gzip
import json
import cv2
import numpy as np
import sys
import torch
from talkingface.model_utils import LoadAudioModel, Audio2bs
from talkingface.data.few_shot_dataset import get_image
from mini_live.render import create_render_model
from talkingface.models.DINet_mini import input_height,input_width
from talkingface.model_utils import device
def interface_mini(path, wav_path, output_video_path):
    # 加载音频模型
    Audio2FeatureModel = LoadAudioModel(r'checkpoint/lstm/lstm_model_epoch_325.pkl')

    # 加载渲染模型
    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("checkpoint/DINet_mini/epoch_40.pth")

    # 设置标准尺寸和裁剪比例
    standard_size = 256
    crop_rotio = [0.5, 0.5, 0.5, 0.5]
    out_w = int(standard_size * (crop_rotio[0] + crop_rotio[1]))
    out_h = int(standard_size * (crop_rotio[2] + crop_rotio[3]))
    out_size = (out_w, out_h)
    renderModel_gl = create_render_model((out_w, out_h), floor=20)

    # 读取 Gzip 压缩的 JSON 文件
    combined_data_path = os.path.join(path, "combined_data.json.gz")
    with gzip.open(combined_data_path, 'rt', encoding='UTF-8') as f:
        combined_data = json.load(f)

    # 从 combined_data 中提取数据
    face3D_obj = combined_data["face3D_obj"]
    json_data = combined_data["json_data"]
    ref_data = np.array(combined_data["ref_data"], dtype=np.float32).reshape([1, 20, input_height//4, input_width//4])

    # 设置 ref_data 到渲染模型
    renderModel_mini.net.infer_model.ref_in_feature = torch.from_numpy(ref_data).float().to(device)

    # 读取视频信息
    video_path = os.path.join(path, "01.mp4")
    cap = cv2.VideoCapture(video_path)
    vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化列表
    list_source_crop_rect = []
    list_video_img = []
    list_standard_img = []
    list_standard_v = []

    # 处理每一帧
    for frame_index in range(min(vid_frame_count, len(json_data))):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        standard_v = json_data[frame_index]["points"][16:]
        source_crop_rect = json_data[frame_index]["rect"]

        standard_img = get_image(frame, source_crop_rect, input_type="image", resize=standard_size)

        list_video_img.append(frame)
        list_source_crop_rect.append(source_crop_rect)
        list_standard_img.append(standard_img)
        list_standard_v.append(np.array(standard_v).reshape(-1, 2) * 2)
    cap.release()

    # 生成矩阵列表
    mat_list = [np.array(i["points"][:16]).reshape(4, 4) * 2 for i in json_data]

    # 反转列表中的数据
    list_video_img_reversed = list_video_img[::-1]
    list_source_crop_rect_reversed = list_source_crop_rect[::-1]
    list_standard_img_reversed = list_standard_img[::-1]
    list_standard_v_reversed = list_standard_v[::-1]
    mat_list_reversed = mat_list[::-1]

    # 将反转后的数据与原有数据合并
    list_video_img = list_video_img + list_video_img_reversed
    list_source_crop_rect = list_source_crop_rect + list_source_crop_rect_reversed
    list_standard_img = list_standard_img + list_standard_img_reversed
    list_standard_v = list_standard_v + list_standard_v_reversed
    mat_list = mat_list + mat_list_reversed

    # 解析 face3D.obj 数据
    v_ = []
    for line in face3D_obj:
        if line.startswith("v "):
            v0, v1, v2, v3, v4 = line[2:].split()
            v_.append(float(v0))
            v_.append(float(v1))
            v_.append(float(v2))
            v_.append(float(v3))
            v_.append(float(v4))
    face_wrap_entity = np.array(v_).reshape(-1, 5)

    # 生成 VBO
    renderModel_gl.GenVBO(face_wrap_entity)

    # 生成音频特征
    bs_array = Audio2bs(wav_path, Audio2FeatureModel)[5:] * 0.5

    # 创建视频写入器
    task_id = str(uuid.uuid1())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "{}.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))

    # 渲染每一帧
    for index2_ in range(len(bs_array)):
        frame_index = index2_ % len(mat_list)
        bs = np.zeros([12], dtype=np.float32)
        bs[:6] = bs_array[frame_index, :6]
        bs[1] = bs[1] / 2 * 1.6

        verts_frame_buffer = np.array(list_standard_v)[frame_index, :, :2].copy() / 256. * 2 - 1

        rgba = renderModel_gl.render2cv(verts_frame_buffer, out_size=out_size, mat_world=mat_list[frame_index],
                                        bs_array=bs)
        rgba = rgba[::2, ::2, :]
        gl_tensor = torch.from_numpy(rgba / 255.).float().permute(2, 0, 1).unsqueeze(0)
        source_tensor = cv2.resize(list_standard_img[frame_index], (128, 128))
        source_tensor = torch.from_numpy(source_tensor / 255.).float().permute(2, 0, 1).unsqueeze(0)

        warped_img = renderModel_mini.interface(source_tensor.to(device), gl_tensor.to(device))

        image_numpy = warped_img.detach().squeeze(0).cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)

        x_min, y_min, x_max, y_max = list_source_crop_rect[frame_index]

        img_face = cv2.resize(image_numpy, (x_max - x_min, y_max - y_min))
        img_bg = list_video_img[frame_index][:, :, :3]
        img_bg[y_min:y_max, x_min:x_max, :3] = img_face[:, :, :3]

        videoWriter.write(img_bg[:, :, ::-1])
    videoWriter.release()

    # 使用 ffmpeg 合并音频和视频
    os.system(
        "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p -y {}".format(save_path, wav_path, output_video_path))
    os.remove(save_path)

    cv2.destroyAllWindows()

def main():
    # 检查命令行参数的数量
    if len(sys.argv) < 4:
        print("Usage: python demo_mini.py <asset_path> <audio_path> <output_video_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取命令行参数
    asset_path = sys.argv[1]
    print(f"Video asset path is set to: {asset_path}")
    wav_path = sys.argv[2]
    print(f"Audio path is set to: {wav_path}")
    output_video_name = sys.argv[3]
    print(f"Output video name is set to: {output_video_name}")

    # 调用主函数
    interface_mini(asset_path, wav_path, output_video_name)

# 示例使用
if __name__ == "__main__":
    main()