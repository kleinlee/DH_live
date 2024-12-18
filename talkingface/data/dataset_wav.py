import torch
import torch.utils.data as data
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, PolarityInversion
# from audio import melspectrogram,mel_bar
import kaldi_native_fbank as knf
import random



class AudioVisualDataset(data.Dataset):
    """ audio-visual dataset. currently, return 2D info and 3D tracking info.

    '''
        多个片段的APC语音特征和嘴部顶点的PCA信息
        :param audio_features: list
        :param mouth_features: list
    '''

    """

    def __init__(self, audio_features, mouth_features, is_train = True, seq_len = 9):
        super(AudioVisualDataset, self).__init__()

        self.fps = 25
        # 每0.2s一个序列
        # self.seq_len = int(self.fps /5)
        self.seq_len = seq_len
        self.frame_jump_stride = 2
        self.audio_features = audio_features
        self.bs_features = mouth_features
        self.is_train = is_train

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PolarityInversion(p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ])


        # 每个序列的裁剪片段个数
        self.clip_num = []
        for i in range(len(audio_features)):
            audio_frame_num = int(len(self.audio_features[i])/(16000/25)) - 2
            self.clip_num.append(min(len(self.bs_features[i]), audio_frame_num) - self.seq_len + 1)

    def __getitem__(self, index):
        if self.is_train:
            video_index = random.randint(0, len(self.bs_features) - 1)
            # print(video_index, self.clip_num[video_index])
            clips_index = random.sample(range(self.clip_num[video_index]), 1)  # 从当前视频选1个片段
            current_frame = clips_index[0]
        else:
            video_index = 0
        # video_index = 0
        # for i in range(len(self.clip_num)):
        #     if index < np.sum(self.clip_num[:i+1]):
        #         video_index = i
        #         break
        # current_frame = index - np.sum(self.clip_num[:video_index], dtype=int)
        # print(video_index, current_frame, current_frame + self.seq_len, self.clip_num, self.bs_features[video_index].shape)

        # start point is current frame
        A2Lsamples = self.audio_features[video_index][current_frame*640: (current_frame + self.seq_len + 2)*640]
        # A2Lsamples = copy.deepcopy(A2Lsamples_)
        # print("A2Lsamples: ", A2Lsamples.shape, A2Lsamples.dtype, A2Lsamples.__class__)
        augmented_samples = self.augment(np.array(A2Lsamples, dtype=np.float32), sample_rate=16000)
        # print(augmented_samples.shape, augmented_samples.dtype)
        # int16转换为float格式
        augmented_samples2 = augmented_samples.astype(np.float32, order='C') / 32768.0
        # orig_mel = mel_bar(augmented_samples2)
        # orig_mel = melspectrogram(augmented_samples2).T
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.frame_length_ms = 50
        opts.frame_opts.frame_shift_ms = 20
        opts.mel_opts.num_bins = 80
        opts.frame_opts.snip_edges = False
        opts.mel_opts.debug_mel = False
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(16000, augmented_samples2.tolist())
        A2Lsamples = np.zeros([2*self.seq_len, 80])
        for i in range(2*self.seq_len):
            f2 = fbank.get_frame(i)
            A2Lsamples[i] = f2
        fbank.input_finished()

        target_bs = self.bs_features[video_index][current_frame: current_frame + self.seq_len, :].reshape(
            self.seq_len, -1)

        # target_bs = self.bs_features[video_index][current_frame + self.seq_len//2, :]

        A2Lsamples = torch.from_numpy(A2Lsamples).float()
        target_bs = torch.from_numpy(target_bs).float()
        # print("*****", A2Lsamples.size(), target_bs.size(), len(self.clip_num))

        return [A2Lsamples, target_bs]

    def __len__(self):
        return len(self.clip_num)
        # return np.sum(self.clip_num, dtype = int)
        # return 10000