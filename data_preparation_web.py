import uuid
import tqdm
import numpy as np
import cv2
import sys
import os
import gzip
from talkingface.data.few_shot_dataset import get_image
import shutil
from talkingface.utils import crop_mouth, main_keypoints_index, smooth_array, normalizeLips, INDEX_LIPS_OUTER
import json
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap
import pickle
from talkingface.models.DINet_mini import model_size


def is_speaking(pts_3d, INDEX_LIPS_OUTER,
                motion_threshold=0.08,
                min_frames_ratio=0.15):
    """
    判断整个视频中人物是否在说话（通过嘴部运动幅度）

    Args:
        pts_3d: [N, 20, 3] 嘴部外围关键点
        INDEX_LIPS_OUTER: 索引列表（长度20，顺时针从左侧嘴角开始）
        motion_threshold: 嘴部开合程度的变化阈值（超过此值视为一次嘴部运动）
        min_frames_ratio: 有明显运动的帧数占总帧数的最小比例

    Returns:
        bool: True表示视频中人物在说话
    """
    outer_points = pts_3d[:, INDEX_LIPS_OUTER]

    # 计算每帧的嘴巴张开比例
    left_corner = outer_points[:, 0, :]
    right_corner = outer_points[:, 10, :]
    mouth_width = np.linalg.norm(right_corner - left_corner, axis=1)

    upper_lip = outer_points[:, 5, :]
    lower_lip = outer_points[:, 15, :]
    vertical_dist = np.linalg.norm(lower_lip - upper_lip, axis=1)

    open_ratio = vertical_dist / (mouth_width + 1e-6)

    # 计算相邻帧之间的变化幅度（嘴部运动速度）
    motion = np.abs(np.diff(open_ratio))

    # 判断每帧是否发生了明显的嘴部运动
    has_motion = np.concatenate([[False], motion > motion_threshold])

    # 统计有运动的帧比例
    if len(has_motion) == 0:
        return False

    motion_frame_ratio = np.mean(has_motion)

    return motion_frame_ratio >= min_frames_ratio

def save_thumbnail(frame_bgr, vid_width, vid_height, output_thumbnail):
    if vid_width > vid_height:
        new_width = 480
        new_height = int((vid_height / vid_width) * 480)
    else:
        new_height = 480
        new_width = int((vid_width / vid_height) * 480)

    resized_frame = cv2.resize(frame_bgr, (new_width, new_height))
    cv2.imwrite(output_thumbnail, resized_frame)

def step0_keypoints(video_path, out_path):
    Path_output_pkl = video_path + "/processed.pkl"
    with open(Path_output_pkl, "rb") as f:
        pts_3d = pickle.load(f)

    pts_3d = pts_3d.reshape(len(pts_3d), -1)
    smooth_array_ = smooth_array(pts_3d, weight=[0.015, 0.095, 0.78, 0.095, 0.015])
    pts_3d = smooth_array_.reshape(len(pts_3d), 478, 3)

    is_open_mouth = is_speaking(pts_3d[:, main_keypoints_index], INDEX_LIPS_OUTER)
    print(f"视频存在明显说话: {is_open_mouth}")
    if is_open_mouth:
        raise ValueError("视频中存在明显说话行为")

    video_path = os.path.join(video_path, "processed.mp4")
    cap = cv2.VideoCapture(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    cap.release()
    out_path = os.path.join(out_path, "01.mp4")
    try:
        # 复制文件
        shutil.copy(video_path, out_path)
        print(f"视频已成功复制到 {out_path}")
    except Exception as e:
        print(f"复制文件时出错: {e}")
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
    face_mid = face_mid.astype(int)
    if face_mid[:, 0].max() + face_size / 2 > vid_width or face_mid[:, 1].max() + face_size / 2 > vid_height:
        raise ValueError("人脸范围超出了视频，请保证视频合格后再重试")

    list_source_crop_rect = np.concatenate([face_mid - face_size // 2, face_mid + face_size // 2], axis = 1)

    standard_size = model_size
    list_standard_v = []
    for frame_index in range(len(list_source_crop_rect)):
        source_pts = pts_3d[frame_index]
        source_crop_rect = list_source_crop_rect[frame_index]
        standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize=standard_size)

        list_standard_v.append(standard_v)

    return list_source_crop_rect, list_standard_v

def generate_combined_data(list_source_crop_rect, list_standard_v, video_path, out_path):
    from mini_live.obj.obj_utils import generateRenderInfo, generateWrapModel
    from talkingface.run_utils import calc_face_mat
    from mini_live.obj.wrap_utils import newWrapModel
    from talkingface.render_model_mini import RenderModel_Mini

    # Step 2: Generate face3D.obj data
    render_verts, render_face = generateRenderInfo()
    face_pts_mean = render_verts[:478, :3].copy()

    wrapModel_verts, wrapModel_face = generateWrapModel()
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v), face_pts_mean)

    face_pts_mean_personal_primer = normalizeLips(face_pts_mean_personal_primer, face_pts_mean)
    face_wrap_entity = newWrapModel(wrapModel_verts, face_pts_mean_personal_primer)

    face3D_data = []
    for i in face_wrap_entity:
        face3D_data.append("v {:.3f} {:.3f} {:.3f} {:.02f} {:.0f}\n".format(i[0], i[1], i[2], i[3], i[4]))
    for i in range(len(wrapModel_face) // 3):
        face3D_data.append("f {0} {1} {2}\n".format(wrapModel_face[3 * i] + 1, wrapModel_face[3 * i + 1] + 1,
                                                    wrapModel_face[3 * i + 2] + 1))

    # Step 3: Generate ref_data.txt data
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("checkpoint/DINet_mini/epoch_40_new.pth")

    Path_output_pkl = "{}/processed.pkl".format(video_path)
    with open(Path_output_pkl, "rb") as f:
        ref_images_info = pickle.load(f)

    video_path = "{}/processed.mp4".format(video_path)
    cap = cv2.VideoCapture(video_path)
    vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert vid_frame_count > 0, "处理后的视频无有效帧"
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    standard_size = model_size
    frame_index = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_bgr = cap.read()
    cap.release()

    thumbnail_img_path = os.path.join(out_path, "thumbnail.jpg")
    save_thumbnail(frame_bgr, vid_width, vid_height, thumbnail_img_path)

    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    source_pts = ref_images_info[frame_index]
    source_crop_rect = crop_mouth(source_pts[main_keypoints_index], vid_width, vid_height)

    standard_img = get_image(frame, source_crop_rect, input_type="image", resize=standard_size)
    standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize=standard_size)

    renderModel_mini.reset_charactor(standard_img, standard_v[main_keypoints_index], standard_size=standard_size)

    ref_in_feature = renderModel_mini.net.infer_model.ref_in_feature
    print(renderModel_mini.net.ref_bg_feature)
    ref_in_feature = ref_in_feature.detach().squeeze(0).cpu().float().numpy().flatten().tolist() + renderModel_mini.net.ref_bg_feature.flatten().tolist()
    # cv2.imwrite(os.path.join(out_path, 'ref.png'), renderModel_mini.ref_img_save)
    rounded_array = np.round(ref_in_feature, 6)

    # Combine all data into a single JSON object
    combined_data = {
        "uid": "matesx_" + str(uuid.uuid4()),
        "frame_num": len(list_standard_v),
        "face3D_obj": face3D_data,
        "ref_data": rounded_array.tolist(),
        "json_data": [],
        "authorized": False,
        "size": model_size,
        "version": 1
    }

    for frame_index in range(len(list_source_crop_rect)):
        source_crop_rect = list_source_crop_rect[frame_index]
        standard_v = list_standard_v[frame_index]

        standard_v = standard_v[index_wrap[:-1], :2].flatten().tolist()
        mat = mat_list[frame_index].T.flatten().tolist()
        standard_v_rounded = [round(i, 5) for i in mat] + [round(i, 1) for i in standard_v]
        combined_data["json_data"].append({"rect": source_crop_rect.tolist(), "points": standard_v_rounded})

    # Save as Gzip compressed JSON
    output_file = os.path.join(out_path, "combined_data.json.gz")
    with gzip.open(output_file, 'wt', encoding='UTF-8') as f:
        json.dump(combined_data, f)

def data_preparation_web(path):
    video_path = os.path.join(path, "data")
    out_path = os.path.join(path, "assets")
    os.makedirs(out_path, exist_ok=True)
    pts_3d, vid_width,vid_height = step0_keypoints(video_path, out_path)
    list_source_crop_rect, list_standard_v = step1_crop_mouth(pts_3d, vid_width, vid_height)
    generate_combined_data(list_source_crop_rect, list_standard_v, video_path, out_path)
    print("Done!")

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 2:
        print("Usage: python data_preparation_web.py <video_dir_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_dir_path = sys.argv[1]

    data_preparation_web(video_dir_path)

if __name__ == "__main__":
    main()
