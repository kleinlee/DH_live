import sys
import numpy as np
import kaldi_native_fbank as knf
from scipy.io import wavfile
import torch
import librosa
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
pca = None
def LoadAudioModel(ckpt_path):
    # if method == "lstm":
    #     ckpt_path = 'checkpoint/lstm/lstm_model_epoch_560.pth'
    #     Audio2FeatureModel = torch.load(model_path).to(device)
    #     Audio2FeatureModel.eval()
    from talkingface.models.audio2bs_lstm import Audio2Feature
    Audio2FeatureModel = Audio2Feature()  # 调用模型Model
    checkpoint = torch.load(ckpt_path, map_location=device)
    Audio2FeatureModel.load_state_dict(checkpoint)
    Audio2FeatureModel = Audio2FeatureModel.to(device)
    Audio2FeatureModel.eval()
    return Audio2FeatureModel

def LoadRenderModel(ckpt_path, model_name = "one_ref"):
    if model_name == "one_ref":
        from talkingface.models.DINet import LeeNet as DINet
        n_ref = 1
        source_channel = 3
        ref_channel = n_ref * 6
    else:
        from talkingface.models.DINet import DINet_five_Ref as DINet
        n_ref = 5
        source_channel = 6
        ref_channel = n_ref * 6
    net_g = DINet(source_channel, ref_channel).to(device)
    checkpoint = torch.load(ckpt_path)
    net_g_static = checkpoint['state_dict']['net_g']
    net_g.load_state_dict(net_g_static)
    net_g.eval()
    return net_g

def find_silence_frame(wav):
    audio_energy = librosa.feature.rms(y=wav, frame_length=800, hop_length=320,
                                       center=False)  # 对每一frame计算root-mean-square
    audio_energy = np.transpose(audio_energy)  # [t,1]
    energy = audio_energy[:2 * (len(audio_energy) // 2)]
    energy = energy.reshape([-1, 2]).max(axis=1)  # downsample with max_pool
    is_sil_mask = (energy < 1e-3).astype(int)
    sil_index_list = []
    for i in range(len(is_sil_mask)):
        if i < 3 or i > len(is_sil_mask) - 4:
            sil_index_list.append(is_sil_mask[i])
        else:
            if is_sil_mask[i] > 0:
                sil_index_list.append(is_sil_mask[i])
            else:
                # 七帧里面有四帧都静音，且左右侧都有静音帧，则此帧判别为静音帧
                if np.sum(is_sil_mask[i-3:i+4]) > 3 and np.sum(is_sil_mask[i-3:i+1]) > 0 and np.sum(is_sil_mask[i+1:i+4]) > 0:
                    sil_index_list.append(1)
                else:
                    sil_index_list.append(0)
    sil_index = np.where(sil_index_list)[0]
    return sil_index


def Audio2mouth(wavpath, Audio2FeatureModel,  method = "lstm"):
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
    # sss = augmented_samples2.tolist()
    # for ii in range(0, len(sss), 10000):
    #     fbank.accept_waveform(16000, sss[ii:ii+10000])
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
    bs_array, hn, cn = Audio2FeatureModel(input, h0, c0)
    # print(bs_array.shape)
    bs_array = bs_array[0].detach().cpu().float().numpy()
    # print(bs_array.shape)
    bs_array = bs_array[4:]
    bs_array[:, :2] = bs_array[:, :2] / 8
    bs_array[:, 2] = - bs_array[:, 2] / 8

    return bs_array
from scipy.signal import resample
def Audio2bs(wavpath, Audio2FeatureModel):
    rate, wav = wavfile.read(wavpath, mmap=False)
    wav = resample(wav, len(wav) //2)
    augmented_samples = wav
    augmented_samples2 = augmented_samples.astype(np.float32, order='C') / 32768.0
    # print(augmented_samples2.shape, augmented_samples2.shape[0] / 16000)

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = 8000
    opts.frame_opts.frame_length_ms = 50
    opts.frame_opts.frame_shift_ms = 20
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False
    fbank = knf.OnlineFbank(opts)
    # sss = augmented_samples2.tolist()
    # for ii in range(0, len(sss), 10000):
    #     fbank.accept_waveform(16000, sss[ii:ii+10000])
    fbank.accept_waveform(8000, augmented_samples2.tolist())
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
    bs_array, hn, cn = Audio2FeatureModel(input, h0, c0)
    # print(bs_array.shape)
    bs_array = bs_array[0].detach().cpu().float().numpy()
    # print(bs_array.shape)
    # bs_array = bs_array[4:]
    return bs_array
