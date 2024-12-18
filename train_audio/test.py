import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))

import pickle
import numpy as np
import kaldi_native_fbank as knf
from scipy.io import wavfile
import torch
import glob
import cv2
import os
from talkingface.models.audio2bs_lstm import Audio2Feature

def main(wavpath, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载模型
    Audio2FeatureModel = Audio2Feature().to(device)
    Audio2FeatureModel.load_state_dict(torch.load(ckpt_path))
    Audio2FeatureModel.eval()
    Path_output_pkl = "checkpoints/pca.pkl"
    with open(Path_output_pkl, "rb") as f:
        pca = pickle.load(f)
    rate, wav = wavfile.read(wavpath, mmap=False)
    augmented_samples = wav
    augmented_samples2 = augmented_samples.astype(np.float32, order='C') / 32768.0
    print(augmented_samples2.shape, augmented_samples2.shape[0] / 16000)

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.frame_length_ms = 50
    opts.frame_opts.frame_shift_ms = 20
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, augmented_samples2.tolist())
    seq_len = fbank.num_frames_ready // 2
    A2Lsamples = np.zeros([2 * seq_len, 80])
    for i in range(2 * seq_len):
        print(i)
        f2 = fbank.get_frame(i)
        A2Lsamples[i] = f2

    orig_mel = A2Lsamples
    print(orig_mel.shape)
    input = torch.from_numpy(orig_mel).unsqueeze(0).float().to(device)
    print(input.shape)
    h0 = torch.zeros(2, 1, 192).to(device)
    c0 = torch.zeros(2, 1, 192).to(device)
    bs_array, hn, cn = Audio2FeatureModel(input, h0, c0)
    print(bs_array.shape)
    bs_array = bs_array[0].detach().cpu().float().numpy()
    print(bs_array.shape)
    bs_array = bs_array[4:]
    frame_num = len(bs_array)
    import uuid

    task_id = str(uuid.uuid1())
    for frame_index in range(frame_num):
        bs_real = bs_array[frame_index]
        pts = np.dot(bs_real[:2], pca.components_[:2]) + pca.mean_
        ref_img_ = pts.reshape(15, -1, 3).astype(np.uint8)
        frame = cv2.resize(ref_img_, (ref_img_.shape[1] * 4, ref_img_.shape[0] * 4))

        # for point in pts:
        #     cv2.circle(frame, (int(point[0]), int(point[1])), point_size, point_color, thickness)

        os.makedirs("output/{}".format(task_id), exist_ok=True)
        cv2.imwrite("output/{}/{:>06d}.png".format(task_id, frame_index), frame)
        # cv2.imshow("aa", frame)
        # cv2.waitKey(16)

    cv2.destroyAllWindows()
    val_video = "output/{}.mp4".format(task_id)
    os.system(
        "ffmpeg -r 25 -i output/{}/%06d.png -i {} -c:v libx264 -pix_fmt yuv420p {}".format(task_id, wavpath, val_video))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test.py <wav_path> <ckpt_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    wav_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    # model_path = 'checkpoints/audio.pkl'
    main(wav_path, ckpt_path)