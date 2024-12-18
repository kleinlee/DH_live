import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
from sklearn import decomposition
import imageio
import numpy as np
import cv2
import pickle
import tqdm
import glob
from talkingface.utils import INDEX_LIPS_OUTER,main_keypoints_index
from talkingface.mediapipe_utils import detect_face_mesh

def main(video_path_list):
    cap = cv2.VideoCapture(video_path_list[0])
    vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(min(500, vid_frame_count)):
        frames.append(cap.read()[1])
    cap.release()
    keypoints = detect_face_mesh(frames)
    keypoints = keypoints[:, main_keypoints_index]
    print(keypoints.shape, len(main_keypoints_index))
    # 先根据一个视频的前500帧估计嘴巴范围
    x_min, x_max = np.min(keypoints[:, INDEX_LIPS_OUTER, 0]), np.max(keypoints[:, INDEX_LIPS_OUTER, 0])
    y_min, y_max = np.min(keypoints[:, INDEX_LIPS_OUTER, 1]), np.max(keypoints[:, INDEX_LIPS_OUTER, 1])
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_len, y_len = (x_max - x_min) / 2, (y_max - y_min) / 2
    x_min, x_max = x_mid - x_len * 0.9, x_mid + x_len * 0.9
    y_min, y_max = y_mid - y_len * 0.9, y_mid + y_len * 0.9
    print("嘴部区域:", x_min, x_max, y_min, y_max)
    frames = []
    num_sum = 0
    for video_path in video_path_list:
        cap = cv2.VideoCapture(video_path)
        vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(vid_frame_count):
            frame = cv2.resize(cap.read()[1][int(y_min):int(y_max), int(x_min):int(x_max)], (30, 15)).astype(np.float32)
            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
            tmp = (frame[:, :15] + frame[:, 15:][:, ::-1]) / 2
            frame = np.concatenate([tmp, tmp[:, ::-1]], axis=1)
            # max_, min_ = np.max(frame), np.min(frame)
            # print(max_, min_)
            frame = (frame - 60) / (180 - 60.) * 255
            frame = frame.clip(0, 255).astype(np.uint8)
            # cv2.imshow("s", cv2.resize(frame, (400,200)))
            # cv2.waitKey(40)
            frames.append(frame)
        cap.release()
        num_sum += vid_frame_count
        if num_sum > 100000:
            break
        # exit()

    out_size = [30, 15]

    x = np.array([frame.flatten() for frame in frames])
    del frames
    n_components = 7
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(x[:150000])
    y = pca.transform(x[:30000])
    print("PCA建模完成")
    frames = []
    for index_ in range(n_components):
        bs_sort = np.sort(y[:, index_])
        y_min_bs = bs_sort[int(0. * len(bs_sort))]
        y_max_bs = bs_sort[int(1 * len(bs_sort)) - 1]
        frames_ = []
        # 从-0.5到1.5，均分为30份
        list_ = [0 + jj * 1. / 30 for jj in range(30)]
        for index__ in list_:
            weight = y_min_bs + index__ * (y_max_bs - y_min_bs)
            frame = pca.mean_ + pca.components_[index_] * weight
            frame = frame.reshape(out_size[1], out_size[0], 3)
            frame = frame.clip(0, 255).astype(np.uint8)
            frame = cv2.resize(frame, (frame.shape[1] * 20, frame.shape[0] * 20))
            # cv2.imshow("s", frame)
            # cv2.waitKey(30)

            frames_.append(frame)
        frames.append(frames_)
    frames_final = []
    for i in range(30):
        img0 = np.concatenate([frames[0][i], frames[1][i], frames[2][i]], axis=1)
        img1 = np.concatenate([frames[3][i], frames[4][i], frames[5][i]], axis=1)
        frame = np.concatenate([img0, img1])
        frames_final.append(frame[:, :, ::-1])

    imageio.mimsave("checkpoints/wav2lip_pca_all.gif", frames_final, 'GIF', duration=0.15)

    Path_output_pkl = r"checkpoints/pca.pkl"
    with open(Path_output_pkl, "wb") as f:
        pickle.dump(pca, f)

    for video_path in tqdm.tqdm(video_path_list[:]):
        frames = []

        cap = cv2.VideoCapture(video_path)
        vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(vid_frame_count):
            frame = cv2.resize(cap.read()[1][int(y_min):int(y_max), int(x_min):int(x_max)], (30, 15))
            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()

        x = np.array([frame.flatten() for frame in frames])
        y = pca.transform(x)
        pca_coefs_path = video_path.replace(".avi", ".txt")
        np.savetxt(pca_coefs_path, y, fmt="%.02f")
    print("Done!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python preparation_step1.py <data_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    data_path = sys.argv[1]
    video_path_list = glob.glob(os.path.join(data_path, "*.avi"))
    main(video_path_list)