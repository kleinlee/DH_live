import numpy as np
import cv2
import tqdm
import copy
from talkingface.utils import *
import glob
import pickle
import torch
import torch.utils.data as data
from talkingface.models.DINet_mini import input_height,input_width
model_size = (256, 256)

def get_image(A_path, crop_coords, input_type, resize= (256, 256)):
    (x_min, y_min, x_max, y_max) = crop_coords
    size = (x_max - x_min, y_max - y_min)

    if input_type == 'mediapipe':
        pose_pts = (A_path - np.array([x_min, y_min])) * resize / size
        return pose_pts[:, :2]
    else:
        img_output = A_path[y_min:y_max, x_min:x_max, :]
        img_output = cv2.resize(img_output, resize)
        return img_output
def generate_input(img, keypoints, is_train = False, mode=["mouth_bias"]):
    # 根据关键点决定正方形裁剪区域
    crop_coords = crop_mouth(keypoints, img.shape[1], img.shape[0], is_train=is_train)
    target_keypoints = get_image(keypoints[:,:2], crop_coords, input_type='mediapipe', resize = model_size)
    target_img = get_image(img, crop_coords, input_type='img', resize = model_size)

    source_img = copy.deepcopy(target_img)
    source_keypoints = target_keypoints

    source_face_egde = draw_mouth_maps(source_keypoints, im_edges = source_img)
    return source_img,target_img,crop_coords

def generate_ref(img, keypoints, is_train=False, teeth = False):
    crop_coords = crop_mouth(keypoints, img.shape[1], img.shape[0], is_train=is_train)
    ref_keypoints = get_image(keypoints, crop_coords, input_type='mediapipe', resize = model_size)
    ref_img = get_image(img, crop_coords, input_type='img', resize = model_size)

    if teeth:
        teeth_mask = np.zeros((model_size[1], model_size[0], 3), np.uint8)  # edge map for all edges
        pts = ref_keypoints[INDEX_LIPS_INNER, :2]
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(teeth_mask, [pts], color=(1, 1, 1))
        # cv2.imshow("s", teeth_mask*255)
        # cv2.waitKey(-1)
        ref_img = ref_img * teeth_mask

    ref_face_edge = draw_mouth_maps(ref_keypoints, size = model_size)
    ref_img = np.concatenate([ref_img, ref_face_edge[:,:,:1]], axis=2)
    return ref_img

def select_ref_index(driven_keypoints, n_ref = 5, ratio = 1/3., ratio2 = 1):
    # 根据嘴巴开合程度，选取开合最大的那一半
    lips_distance = np.linalg.norm(
        driven_keypoints[:, INDEX_LIPS_INNER[5]] - driven_keypoints[:, INDEX_LIPS_INNER[-5]], axis=1)
    selected_index_list = np.argsort(lips_distance).tolist()[int(len(lips_distance) * ratio): int(len(lips_distance) * ratio2)]
    ref_img_index_list = random.sample(selected_index_list, n_ref)  # 从当前视频选n_ref个图片
    return ref_img_index_list

def get_ref_images_fromVideo(cap, ref_img_index_list, ref_keypoints):
    ref_img_list = []
    for index in ref_img_index_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # 设置要获取的帧号
        ret, frame = cap.read()
        frame = frame[:,:,::-1]
        if ret is False:
            print("请检查当前视频， 错误帧数：", index)
        ref_img = generate_ref(frame, ref_keypoints[index])
        ref_img_list.append(ref_img)
    ref_img = np.concatenate(ref_img_list, axis=2)
    return ref_img




class Few_Shot_Dataset(data.Dataset):
    def __init__(self, dict_info, n_ref = 2, is_train = False):
        super(Few_Shot_Dataset, self).__init__()
        self.driven_images = dict_info["driven_images"]
        self.driven_keypoints = dict_info["driven_keypoints"]
        self.driving_keypoints = dict_info["driving_keypoints"]

        self.driven_teeth_images = dict_info["driven_teeth_image"]
        self.driven_teeth_rect = dict_info["driven_teeth_rect"]
        self.is_train = is_train

        assert len(self.driven_images) == len(self.driven_keypoints)
        assert len(self.driven_images) == len(self.driving_keypoints)

        self.out_size = (256, 256)

        self.sample_num = np.sum([len(i) for i in self.driven_images])

        # list: 每个视频序列的视频块个数
        self.clip_count_list = []  # number of frames in each sequence
        for path in self.driven_images:
            self.clip_count_list.append(len(path))
        self.n_ref = n_ref

    def get_ref_images(self, video_index, ref_img_index_list):
        # 参考图片
        self.ref_img_list = []
        for index_,ref_img_index in enumerate(ref_img_index_list):
            ref_img = cv2.imread(self.driven_images[video_index][ref_img_index])[:, :, ::-1]
            # ref_img = cv2.convertScaleAbs(ref_img, alpha=self.alpha, beta=self.beta)
            ref_keypoints = self.driven_keypoints[video_index][ref_img_index]
            if index_ > 0:
                ref_img = generate_ref(ref_img, ref_keypoints, self.is_train, teeth = True)
            else:
                ref_img = generate_ref(ref_img, ref_keypoints, self.is_train)

            self.ref_img_list.append(ref_img)
        # self.ref_img = np.concatenate(ref_img_list, axis=2)

    def __getitem__(self, index):
        if self.is_train:
            video_index = random.randint(0, len(self.driven_images) - 1)
            current_clip = random.randint(0, self.clip_count_list[video_index] - 1)

            ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref = self.n_ref - 1, ratio = 0.33)      # 从当前视频选n_ref个图片
            ref_img_index_list = [random.randint(0, self.clip_count_list[video_index] - 1)] + ref_img_index_list

            self.get_ref_images(video_index, ref_img_index_list)
        else:
            video_index = 0
            current_clip = index

            if index == 0:
                ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref=self.n_ref)  # 从当前视频选n_ref个图片
                self.get_ref_images(video_index, ref_img_index_list)

        # target图片
        target_img = cv2.imread(self.driven_images[video_index][current_clip])[:, :, ::-1]

        if self.is_train:
            # 统一生成随机参数
            alpha = np.random.uniform(0.8, 1.2)  # 对比度
            beta = 0  # 亮度
            h_shift = np.random.randint(-15, 15)  # 色相偏移

            target_img = cv2.convertScaleAbs(target_img, alpha=alpha, beta=beta)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2HSV)
            target_img[..., 0] = (target_img[..., 0] + h_shift) % 180
            target_img = cv2.cvtColor(target_img, cv2.COLOR_HSV2RGB)

            for ii in range(len(self.ref_img_list)):
                ref_img = self.ref_img_list[ii][:,:,:3]
                ref_img = cv2.convertScaleAbs(ref_img, alpha=alpha, beta=beta)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
                ref_img[..., 0] = (ref_img[..., 0] + h_shift) % 180
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_HSV2RGB)
                self.ref_img_list[ii] = np.concatenate([ref_img, self.ref_img_list[ii][:,:,3:]], axis = 2)

        self.ref_img = np.concatenate(self.ref_img_list, axis=2)

        # target_img = cv2.convertScaleAbs(target_img, alpha=self.alpha, beta=self.beta)
        ref_face_edge = np.zeros_like(target_img)
        target_keypoints = self.driving_keypoints[video_index][current_clip]
        source_img, target_img,crop_coords = generate_input(target_img, target_keypoints, self.is_train, mode="mouth")


        [x_min, y_min, x_max, y_max] = self.driven_teeth_rect[video_index][current_clip]
        teeth_img = cv2.imread(self.driven_teeth_images[video_index][current_clip])
        # print(ref_face_edge.shape, crop_coords, [x_min, y_min, x_max, y_max])
        ref_face_edge[int(y_min):int(y_max), int(x_min):int(x_max), 1][
            np.where(teeth_img[:, teeth_img.shape[1] // 2:, 0] == 0)] = 255
        ref_face_edge[int(y_min):int(y_max), int(x_min):int(x_max), 2][
            np.where(teeth_img[:, teeth_img.shape[1] // 2:, 0] == 255)] = 255


        # cv2.imshow("s", ref_face_edge)
        # cv2.waitKey(-1)
        # print(ref_face_edge.shape, crop_coords)
        teeth_img = get_image(ref_face_edge, crop_coords, input_type='img', resize = model_size)


        source_img[:,:,1][np.where(source_img[:,:,0] == 0)] = teeth_img[:,:,1][np.where(source_img[:,:,0] == 0)]
        source_img[:, :, 2][np.where(source_img[:, :, 0] == 0)] = teeth_img[:, :, 2][np.where(source_img[:, :, 0] == 0)]

        target_img = cv2.resize(target_img, (128, 128))
        source_img = cv2.resize(source_img, (128, 128))
        self.ref_img = cv2.resize(self.ref_img, (128, 128))


        w_pad = int((128 - input_width) / 2)
        h_pad = int((128 - input_height) / 2)

        target_img = target_img[h_pad:-h_pad, w_pad:-w_pad]/255.
        source_img = source_img[h_pad:-h_pad, w_pad:-w_pad]/255.
        ref_img = self.ref_img[h_pad:-h_pad, w_pad:-w_pad]/255.

        # tensor
        source_tensor = torch.from_numpy(source_img).float().permute(2, 0, 1)
        ref_tensor = torch.from_numpy(ref_img).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_img).float().permute(2, 0, 1)
        return source_tensor, ref_tensor, target_tensor, self.driven_images[video_index][current_clip]

    def __len__(self):
        if self.is_train:
            return len(self.driven_images)
        else:
            return len(self.driven_images[0])
        # return self.sample_num
def data_preparation(train_video_list):
    img_all = []
    keypoints_all = []
    teeth_img_all = []
    teeth_rect_all = []
    for i in tqdm.tqdm(train_video_list):
        # for i in ["xiaochangzhang/00004"]:
        model_name = i
        img_filelist = glob.glob("{}/image/*.png".format(model_name))
        img_filelist.sort()
        if len(img_filelist) == 0:
            continue
        img_teeth_filelist = glob.glob("{}/teeth_seg/*.png".format(model_name))
        img_teeth_filelist.sort()

        teeth_rect_array = np.loadtxt("{}/teeth_seg/all.txt".format(model_name))
        Path_output_pkl = "{}/keypoint_rotate.pkl".format(model_name)
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)

        # print(len(img_filelist), len(images_info), len(img_teeth_filelist), len(teeth_rect_array))
        # exit(1)
        valid_frame_num = min(len(img_filelist), len(images_info), len(img_teeth_filelist), len(teeth_rect_array))

        img_all.append(img_filelist[:valid_frame_num])
        keypoints_all.append(images_info[:valid_frame_num, main_keypoints_index, :2])
        teeth_img_all.append(img_teeth_filelist[:valid_frame_num])
        teeth_rect_all.append(teeth_rect_array[:valid_frame_num])

    print("train size: ", len(img_all))
    dict_info = {}
    dict_info["driven_images"] = img_all
    dict_info["driven_keypoints"] = keypoints_all
    dict_info["driving_keypoints"] = keypoints_all
    dict_info["driven_teeth_rect"] = teeth_rect_all
    dict_info["driven_teeth_image"] = teeth_img_all
    return dict_info
