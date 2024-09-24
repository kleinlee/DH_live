import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
from talkingface.models.common.Discriminator import Discriminator
from talkingface.models.common.VGG19 import Vgg19
from talkingface.models.DINet import DINet_five_Ref
from talkingface.util.utils import GANLoss,get_scheduler, update_learning_rate
from talkingface.config.config import DINetTrainingOptions
from torch.utils.tensorboard import SummaryWriter
from talkingface.util.log_board import log
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import sys
import torch.nn.functional as F
import cv2
from talkingface.data.few_shot_dataset import Few_Shot_Dataset,data_preparation

def Tensor2img(tensor_, channel_index):
    frame = tensor_[channel_index:channel_index + 3, :, :].detach().squeeze(0).cpu().float().numpy()
    frame = np.transpose(frame, (1, 2, 0)) * 255.0
    frame = frame.clip(0, 255)
    return frame.astype(np.uint8)

if __name__ == "__main__":
    '''
    training code of person image generation
    '''
    # load config
    opt = DINetTrainingOptions().parse_args()
    n_ref = 5
    opt.source_channel = 3 * 2
    opt.target_channel = 3
    opt.ref_channel = n_ref * 3 * 2
    opt.batch_size = 4
    opt.result_path = "checkpoint/Dinet_five_ref"
    opt.resume = False
    opt.resume_path = None

    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)


    video_list = []
    # path_ = r"../preparation_bilibili"
    path_ = opt.train_data
    video_list += [os.path.join(path_, i) for i in os.listdir(path_)]

    print("video_selected final: ", len(video_list))
    video_list.sort()
    train_dict_info = data_preparation(video_list[:])
    train_set = Few_Shot_Dataset(train_dict_info, n_ref=n_ref, is_train=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    train_log_path = "train_log.txt"
    train_data_length = len(training_data_loader)
    # init network
    net_g = DINet_five_Ref(opt.source_channel,opt.ref_channel).cuda()
    net_d = Discriminator(opt.target_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()

    # set optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr_d)

    if opt.resume:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        # opt.start_epoch = checkpoint['epoch']
        # opt.start_epoch = 200
        net_g_static = checkpoint['state_dict']['net_g']
        net_g.load_state_dict(net_g_static)
        net_d.load_state_dict(checkpoint['state_dict']['net_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer']['net_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer']['net_d'])

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_d_scheduler = get_scheduler(optimizer_d, opt.non_decay, opt.decay)



    train_log_path = os.path.join("checkpoint/{}/log".format("DiNet_five_ref"), "train")
    os.makedirs(train_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)

    # start train
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        avg_loss_g_perception = 0
        avg_Loss_DI = 0
        avg_Loss_GI = 0
        for iteration, data in enumerate(training_data_loader):
            # read data
            source_tensor, ref_tensor, target_tensor = data
            source_tensor = source_tensor.float().cuda()
            ref_tensor = ref_tensor.float().cuda()
            target_tensor = target_tensor.float().cuda()

            source_tensor, source_prompt_tensor = source_tensor[:, :3], source_tensor[:, 3:]
            # network forward
            fake_out = net_g(source_tensor, source_prompt_tensor, ref_tensor)
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(target_tensor, scale_factor=0.5, mode='bilinear')
            # (1) Update D network
            optimizer_d.zero_grad()
            # compute fake loss
            _,pred_fake_d = net_d(fake_out)
            loss_d_fake = criterionGAN(pred_fake_d, False)
            # compute real loss
            _,pred_real_d = net_d(target_tensor)
            loss_d_real = criterionGAN(pred_real_d, True)
            # Combine D loss
            loss_dI = (loss_d_fake + loss_d_real) * 0.5
            loss_dI.backward(retain_graph=True)
            optimizer_d.step()
            # (2) Update G network
            _, pred_fake_dI = net_d(fake_out)
            optimizer_g.zero_grad()
            # compute perception loss
            perception_real = net_vgg(target_tensor)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(target_tensor_half)
            perception_fake_half = net_vgg(fake_out_half)
            loss_g_perception = 0
            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
            loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception
            # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)
            # combine perception loss and gan loss
            loss_g = loss_g_perception + loss_g_dI
            loss_g.backward()
            optimizer_g.step()
            message = "===> Epoch[{}]({}/{}): Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_perception: {:.4f} lr_g = {:.7f} lr_d = {:.7f}".format(
                    epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),
                    float(loss_g_perception), optimizer_g.param_groups[0]['lr'], optimizer_d.param_groups[0]['lr'])
            print(message)
            # with open("train_log.txt", "a") as f:
            #     f.write(message + "\n")

            if iteration%200 == 0:
                inference_out = fake_out * 255
                inference_out = inference_out[0].cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
                inference_in = (target_tensor[0, :3]* 255).cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
                inference_in_prompt = (source_prompt_tensor[0, :3] * 255).cpu().permute(1, 2, 0).float().detach().numpy().astype(
                    np.uint8)
                frame2 = Tensor2img(ref_tensor[0], 0)
                frame3 = Tensor2img(ref_tensor[0], 3)
                inference_out = np.concatenate([inference_in, inference_in_prompt, inference_out, frame2, frame3], axis=1)
                inference_out = cv2.cvtColor(inference_out, cv2.COLOR_RGB2BGR)

                log(train_logger, fig=inference_out, tag="Training/epoch_{}_{}".format(epoch, iteration))

                real_iteration = epoch * len(training_data_loader) + iteration
                message1 = "Step {}/{}, ".format(real_iteration, (epoch + 1) * len(training_data_loader))
                message2 = ""
                losses = [loss_dI.item(), loss_g_perception.item(), loss_g_dI.item()]
                train_logger.add_scalar("Loss/loss_dI", losses[0], real_iteration)
                train_logger.add_scalar("Loss/loss_g_perception", losses[1], real_iteration)
                train_logger.add_scalar("Loss/loss_g_dI", losses[2], real_iteration)

            avg_loss_g_perception += loss_g_perception.item()
            avg_Loss_DI += loss_dI.item()
            avg_Loss_GI += loss_g_dI.item()
        train_logger.add_scalar("Loss/{}".format("epoch_g_perception"), avg_loss_g_perception / len(training_data_loader), epoch)
        train_logger.add_scalar("Loss/{}".format("epoch_DI"),
                                avg_Loss_DI / len(training_data_loader), epoch)
        train_logger.add_scalar("Loss/{}".format("epoch_GI"),
                                avg_Loss_GI / len(training_data_loader), epoch)
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # checkpoint
        if epoch % opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(), 'net_d': net_d.state_dict()},
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_d': optimizer_d.state_dict()}
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))