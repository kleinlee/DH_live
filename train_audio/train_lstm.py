import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
import glob

model_name = "lstm"
seq_len = 26
from talkingface.models.audio2bs_lstm import Audio2Feature
import torch
from torch.utils.data import DataLoader
from talkingface.data.dataset_wav import AudioVisualDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
from scipy.io import wavfile
import pickle
import cv2
import tqdm
import numpy as np
import os


def main(data_path):
    pcm_16k_list = []
    pca_info_list = []
    print("Loading all data into memory. Make sure your memory is enough.")
    wav2_dir = glob.glob(os.path.join(data_path, "*.wav"))
    for index_, wav_file in enumerate(wav2_dir):
        coef_path = wav_file.replace(".wav", ".txt")
        pca_coef = np.loadtxt(coef_path)
        rate, wav = wavfile.read(wav_file, mmap=False)
        pca_info_list.append(pca_coef[:, :6])
        pcm_16k_list.append(wav)
    print("All data Loaded. Total wav files number: {}. Total pca files number: {}.".format(len(pcm_16k_list),
                                                                                            len(pca_info_list)))
    Path_output_pkl = "checkpoints/pca.pkl"
    with open(Path_output_pkl, "rb") as f:
        pca = pickle.load(f)

    def generate_ref_image(bs_real, bs_pred):
        # print(bs_real.shape, bs_pred.shape)
        pts = np.dot(bs_real[:8], pca.components_[:6]) + pca.mean_
        ref_img_ = pts.reshape(8, 15, 30, 3).transpose(1, 0, 2, 3).reshape(15, -1, 3).astype(np.uint8)
        ref_img_ = cv2.resize(ref_img_, (ref_img_.shape[1] * 5, ref_img_.shape[0] * 5))
        pts = np.dot(bs_pred[:8], pca.components_[:6]) + pca.mean_
        ref_img_2 = pts.reshape(8, 15, 30, 3).transpose(1, 0, 2, 3).reshape(15, -1, 3).astype(np.uint8)
        ref_img_2 = cv2.resize(ref_img_2, (ref_img_2.shape[1] * 5, ref_img_2.shape[0] * 5))
        ref_img_ = np.concatenate([ref_img_, ref_img_2], axis=0)
        return ref_img_

    from sklearn.model_selection import train_test_split
    import random
    random_st = random.choice(range(10000))
    print("random_st:", random_st)
    random_st = 777
    train_pcm_16k_list, test_pcm_16k_list, train_pca_info_list, test_pca_info_list = train_test_split(pcm_16k_list,
                                                                                                      pca_info_list,
                                                                                                      test_size=0.2,
                                                                                                      random_state=random_st)

    train_audioVisualDataset = AudioVisualDataset(train_pcm_16k_list, train_pca_info_list, seq_len=seq_len)
    test_audioVisualDataset = AudioVisualDataset(test_pcm_16k_list, test_pca_info_list, seq_len=seq_len)

    training_data_loader = DataLoader(dataset=train_audioVisualDataset, num_workers=0, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(dataset=test_audioVisualDataset, num_workers=0, batch_size=8, shuffle=True)

    # 加载模型
    Audio2FeatureModel = Audio2Feature().to(device)
    criterionL1 = torch.nn.L1Loss().to(device)

    # setup optimizer
    optimizer = torch.optim.Adam(Audio2FeatureModel.parameters(), lr=0.0004, betas=(0.9, 0.99))

    # tensorboard设置
    train_log_path = os.path.join("checkpoints/log", model_name, "train")
    val_log_path = os.path.join("checkpoints/log", model_name, "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    def log(
            logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag="", video=None
    ):
        if losses is not None:
            logger.add_scalar("Loss/loss", losses[0], step)

        if fig is not None:
            logger.add_image(tag, fig, 2, dataformats='HWC')
        if video is not None:
            logger.add_video(tag, video, fps=60)

        if audio is not None:
            logger.add_audio(
                tag,
                audio / max(abs(audio)),
                sample_rate=sampling_rate,
            )

    point_size = 3
    point_color = (0, 0, 255)  # BGR
    thickness = 3  # 0 、4、8
    loss_iteration_100_train = 0
    loss_iteration_100_test = 0
    for epoch in range(0, 1001):
        Audio2FeatureModel.train()
        for iteration, batch in enumerate(training_data_loader):
            A2Lsamples, target_pts2d = batch
            A2Lsamples, target_pts2d = A2Lsamples.to(device), target_pts2d.to(device)
            # print(len(batch), batch[0].size())
            # # A2Lsamples, target_pts2d = batch[0].to(device), batch[1].to(device)
            # print(iteration, A2Lsamples.shape, target_pts2d.shape)
            bs = A2Lsamples.size()[0]
            h0 = torch.zeros(2, bs, 192).to(device)
            c0 = torch.zeros(2, bs, 192).to(device)
            pred_pts2d, _, _ = Audio2FeatureModel(A2Lsamples, h0, c0)

            # print(iteration, A2Lsamples.shape, target_pts2d.shape, pred_pts2d.shape)

            # 25帧的序列，生成25帧的关键点，只使用第5-24帧的结果，和目标关键点序列target_pts2d的第0-19帧对应
            output_dim_size = pred_pts2d.size()
            if len(output_dim_size) > 2:
                pred_pts2d = pred_pts2d[:, 4:]
                target_pts2d = target_pts2d[:, :-4]
            else:
                pred_pts2d = pred_pts2d.view(output_dim_size[0], 1, output_dim_size[1])
                target_pts2d = target_pts2d[:, seq_len // 2:seq_len // 2 + 1]

            # print(pred_pts2d.shape, target_pts2d.shape)
            # exit()

            # Backward and optimize
            optimizer.zero_grad()
            # print(target_pts2d.size(), pred_pts2d.size())
            # pred_pts2d = torch.ones_like(pred_pts2d).cuda() * 0.5
            loss = criterionL1(target_pts2d, pred_pts2d)
            # loss = cosine_loss(target_pts2d, pred_pts2d, bs)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss.item()))
            loss_iteration_100_train += loss.item()

            real_iteration = epoch * len(training_data_loader) + iteration

            if real_iteration % 100 == 0 and real_iteration > 0:
                losses = [loss_iteration_100_train / 100]
                message1 = "Step {}/{}, ".format(real_iteration, epoch * len(training_data_loader))
                message2 = "Loss: {:.4f},".format(
                    *losses
                )

                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")

                # outer_bar.write(message1 + message2)

                log(train_logger, real_iteration, losses=losses)
                loss_iteration_100_train = 0

            if real_iteration % 300 == 0:
                bs_pred = pred_pts2d[0].detach().cpu().float().numpy()
                bs_real = target_pts2d[0].detach().cpu().float().numpy()
                frame = generate_ref_image(bs_real, bs_pred)
                log(
                    train_logger,
                    fig=frame,
                    tag="Training/epoch_{}_{}_predict".format(epoch, iteration),
                )

        Audio2FeatureModel.eval()

        for iteration, batch in enumerate(test_data_loader):
            A2Lsamples, target_pts2d = batch[0].to(device), batch[1].to(device)
            bs = A2Lsamples.size()[0]
            h0 = torch.zeros(2, bs, 192).to(device)
            c0 = torch.zeros(2, bs, 192).to(device)
            pred_pts2d, _, _ = Audio2FeatureModel(A2Lsamples, h0, c0)

            output_dim_size = pred_pts2d.size()
            if len(output_dim_size) > 2:
                pred_pts2d = pred_pts2d[:, 4:]
                target_pts2d = target_pts2d[:, :-4]
            else:
                pred_pts2d = pred_pts2d.view(output_dim_size[0], 1, output_dim_size[1])
                target_pts2d = target_pts2d[:, seq_len // 2:seq_len // 2 + 1]

            # print(iteration, A2Lsamples.shape, target_pts2d.shape, pred_pts2d.shape)
            # pred_pts2d = torch.ones_like(pred_pts2d).cuda() * 0.5
            loss = criterionL1(target_pts2d, pred_pts2d)
            # loss = cosine_loss(target_pts2d, pred_pts2d, bs)
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, iteration, len(test_data_loader), loss.item()))

            real_iteration = epoch * len(test_data_loader) + iteration
            loss_iteration_100_test += loss.item()

            if real_iteration % 100 == 0 and real_iteration > 0:
                losses = [loss_iteration_100_test / 100]
                message1 = "Step {}/{}, ".format(real_iteration, epoch * len(test_data_loader))
                message2 = "Loss: {:.4f},".format(
                    *losses
                )

                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")

                # outer_bar.write(message1 + message2)

                log(val_logger, real_iteration, losses=losses)
                loss_iteration_100_test = 0

            if real_iteration % 300 == 0:
                bs_pred = pred_pts2d[0].detach().cpu().float().numpy()
                bs_real = target_pts2d[0].detach().cpu().float().numpy()
                frame = generate_ref_image(bs_real, bs_pred)
                log(
                    train_logger,
                    fig=frame,
                    tag="Testing/epoch_{}_{}_predict".format(epoch, iteration),
                )
        # checkpoint
        if epoch % 5 == 0:
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")
            if not os.path.exists(os.path.join("checkpoints", model_name)):
                os.mkdir(os.path.join("checkpoints", model_name))
            model_out_path = "checkpoints/{}/epoch_{}.pth".format(model_name, epoch)
            # states = { 'epoch': epoch + 1, 'state_dict': Audio2FeatureModel.state_dict(), 'optimizer': optimizer.state_dict() }
            states = Audio2FeatureModel.state_dict()
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
    torch.save(Audio2FeatureModel.state_dict(), "checkpoints/audio.pkl")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_lstm.py <data_path>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    data_path = sys.argv[1]
    main(data_path)