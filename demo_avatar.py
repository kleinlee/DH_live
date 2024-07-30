import time
import os
import numpy as np
from scipy.io import wavfile
import cv2
import glob
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel

audioModel = AudioModel()
audioModel.loadModel("checkpoint/audio.pkl")

renderModel = RenderModel()
renderModel.loadModel("checkpoint/render.pth")
test_video = "test"
pkl_path = "video_data/{}/keypoint_rotate.pkl".format(test_video)
video_path = "video_data/{}/circle.mp4".format(test_video)
renderModel.reset_charactor(video_path, pkl_path)

wavpath = "video_data/audio0.wav"
rate, wav = wavfile.read(wavpath, mmap=False)
index_ = 0
frame_index__ = 0
import sounddevice as sd
sample_rate = 16000
samples_per_read = int(0.04 * sample_rate)
with sd.InputStream(
        channels=1, dtype="float32", samplerate=sample_rate
) as s:
    while True:
        samples, _ = s.read(samples_per_read)  # a blocking read
        pcm_data = samples.reshape(-1)
        print(pcm_data.shape)
        mouth_frame = audioModel.interface_frame(pcm_data)
        frame = renderModel.interface(mouth_frame)
        cv2.imshow("s", frame)
        cv2.waitKey(10)
        index_ += 1
