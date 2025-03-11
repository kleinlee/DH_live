import sys
import numpy as np
import kaldi_native_fbank as knf
from scipy.io import wavfile
import torch
import pickle
from model_utils import device
import pickle
import os
def pca_process(x):
    a = x.reshape(15, 30, 3)
    # a = pca.mean_.reshape(15,30,3)
    tmp = a[:, :15] + a[:, 15:][:, ::-1]
    a[:, :15] = tmp / 2
    a[:, 15:] = a[:, :15][:, ::-1]
    return a.flatten()
class AudioModel:
    def __init__(self):
        self.__net = None
        self.__fbank = None
        self.__fbank_processed_index = 0
        self.frame_index = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        Path_output_pkl = os.path.join(current_dir, "../data/pca.pkl")
        with open(Path_output_pkl, "rb") as f:
            pca = pickle.load(f)
        self.pca_mean_ = pca_process(pca.mean_)
        self.pca_components_ = np.zeros_like(pca.components_)
        self.pca_components_[0] = pca_process(pca.components_[0])
        self.pca_components_[1] = pca_process(pca.components_[1])
        self.pca_components_[2] = pca_process(pca.components_[2])
        self.pca_components_[3] = pca_process(pca.components_[3])
        self.pca_components_[4] = pca_process(pca.components_[4])
        self.pca_components_[5] = pca_process(pca.components_[5])

        self.reset()

    def loadModel(self, ckpt_path):
        # if method == "lstm":
        #     ckpt_path = 'checkpoint/lstm/lstm_model_epoch_560.pth'
        #     Audio2FeatureModel = torch.load(model_path).to(device)
        #     Audio2FeatureModel.eval()
        from talkingface.models.audio2bs_lstm import Audio2Feature
        self.__net = Audio2Feature()  # 调用模型Model
        self.__net.load_state_dict(torch.load(ckpt_path))
        self.__net = self.__net.to(device)
        self.__net.eval()

    def reset(self):
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.frame_length_ms = 50
        opts.frame_opts.frame_shift_ms = 20
        opts.mel_opts.num_bins = 80
        opts.frame_opts.snip_edges = False
        opts.mel_opts.debug_mel = False
        self.__fbank = knf.OnlineFbank(opts)

        self.h0 = torch.zeros(2, 1, 192).to(device)
        self.c0 = torch.zeros(2, 1, 192).to(device)

        self.__fbank_processed_index = 0

        audio_samples = np.zeros([320])
        self.__fbank.accept_waveform(16000, audio_samples.tolist())

    def interface_frame(self, audio_samples):
        # pcm为uint16位数据。 只处理一帧的数据， 16000/25 = 640
        self.__fbank.accept_waveform(16000, audio_samples.tolist())
        orig_mel = np.zeros([2, 80])

        orig_mel[0] = self.__fbank.get_frame(self.__fbank_processed_index)
        orig_mel[1] = self.__fbank.get_frame(self.__fbank_processed_index + 1)

        input = torch.from_numpy(orig_mel).unsqueeze(0).float().to(device)
        bs_array, self.h0, self.c0 = self.__net(input, self.h0, self.c0)
        bs_array = bs_array[0].detach().cpu().float().numpy()
        bs_real = bs_array[0]
        # print(self.__fbank_processed_index, self.__fbank.num_frames_ready, bs_real)

        frame = np.dot(bs_real[:6], self.pca_components_[:6]) + self.pca_mean_
        # print(frame_index, frame.shape)
        frame = frame.reshape(15, 30, 3).clip(0, 255).astype(np.uint8)
        self.__fbank_processed_index += 2
        return frame

    def interface_wav(self, wavpath):
        rate, wav = wavfile.read(wavpath, mmap=False)
        augmented_samples = wav
        augmented_samples2 = augmented_samples.astype(np.float32, order='C') / 32768.0
        # print(augmented_samples2.shape, augmented_samples2.shape[0] / 16000)

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
            f2 = fbank.get_frame(i)
            A2Lsamples[i] = f2

        orig_mel = A2Lsamples
        # print(orig_mel.shape)
        input = torch.from_numpy(orig_mel).unsqueeze(0).float().to(device)
        # print(input.shape)
        h0 = torch.zeros(2, 1, 192).to(device)
        c0 = torch.zeros(2, 1, 192).to(device)
        bs_array, hn, cn = self.__net(input, h0, c0)
        bs_array = bs_array[0].detach().cpu().float().numpy()
        bs_array = bs_array[4:]

        frame_num = len(bs_array)
        output = np.zeros([frame_num, 15, 30, 3], dtype = np.uint8)
        for frame_index in range(frame_num):
            bs_real = bs_array[frame_index]
            # bs_real[1:4] = - bs_real[1:4]
            frame = np.dot(bs_real[:6], self.pca_components_[:6]) + self.pca_mean_
            # print(frame_index, frame.shape)
            frame = frame.reshape(15, 30, 3).clip(0, 255).astype(np.uint8)
            output[frame_index] = frame

        return output