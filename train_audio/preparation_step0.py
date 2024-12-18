import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))

import sys, audio
from tqdm import tqdm
import glob
from models import Wav2Lip
import cv2
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "checkpoints/wav2lip.pth"

mel_step_size = 16
wav2lip_batch_size = 8
img_size = 96
fps = 25
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
Wav2Lip_model = None
def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def face_detect(image):
    from talkingface.mediapipe_utils import detect_face_mesh
    keypoints = detect_face_mesh([image])[0]
    x_min, y_min, x_max, y_max = np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(
        keypoints[:, 1])
    y_min = y_min - (y_max - y_min) * 0.1
    (x1, y1, x2, y2) = np.array([x_min, y_min, x_max, y_max], dtype = int)
    face = image[y1: y2, x1:x2]
    face = cv2.resize(face, (96, 96))
    return face

def datagen(image, mels):
    img_batch = []
    mel_batch = []
    for i, m in enumerate(mels):
        img_batch.append(image)
        mel_batch.append(m)
        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch
            img_batch, mel_batch = [], []


def main(face_image, wav_path, outfile):
    global Wav2Lip_model
    if Wav2Lip_model is None:
        Wav2Lip_model = load_model(checkpoint_path)
    print(wav_path)
    wav = audio.load_wav(wav_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    batch_size = 8
    gen = datagen(face_image, mel_chunks)

    for i, (img_batch, mel_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'DIVX'), fps, (96, 96))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = Wav2Lip_model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p in pred:
            f = p.astype(np.uint8)
            out.write(f)
    out.release()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preparation_step0.py <face_path> <wav_16K_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    face_path = sys.argv[1]
    print(f"Face path is set to: {face_path}")
    wav_16K_path = sys.argv[2]
    print(f"Wav 16K path is set to: {wav_16K_path}, please make sure all wav files duration not less than 2s.")

    image = cv2.imread(face_path)
    face_pix96 = face_detect(image)
    wav_files = glob.glob(os.path.join(wav_16K_path, "*.wav"))
    for index_, wav_path in enumerate(wav_files):
        outfile = wav_path.replace(".wav", ".avi")
        main(face_pix96, wav_path, outfile)
