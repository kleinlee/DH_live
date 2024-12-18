import numpy as np
import cv2
import tqdm
import copy
from talkingface.utils import *
import glob
import pickle
import torch
import torch.utils.data as data
def get_image(A_path, crop_coords, input_type, resize= 256):
    (x_min, y_min, x_max, y_max) = crop_coords
    size = (x_max - x_min, y_max - y_min)

    if input_type == 'mediapipe':
        if A_path.shape[1] == 2:
            pose_pts = (A_path - np.array([x_min, y_min])) * resize / size
            return pose_pts[:, :2]
        else:
            A_path[:, 2] = A_path[:, 2] - np.max(A_path[:, 2])
            pose_pts = (A_path - np.array([x_min, y_min, 0])) * resize / size[0]
            return pose_pts[:, :3]

    else:
        img_output = A_path[y_min:y_max, x_min:x_max, :]
        img_output = cv2.resize(img_output, (resize, resize))
        return img_output
def generate_input(img, keypoints, mask_keypoints, is_train = False, mode=["mouth_bias"], mouth_width = None, mouth_height = None):
    # 根据关键点决定正方形裁剪区域
    crop_coords = crop_face(keypoints, size=img.shape[:2], is_train=is_train)
    target_keypoints = get_image(keypoints[:,:2], crop_coords, input_type='mediapipe')
    target_img = get_image(img, crop_coords, input_type='img')

    target_mask_keypoints = get_image(mask_keypoints[:,:2], crop_coords, input_type='mediapipe')

    # source_img信息：扣出嘴部区域
    source_img = copy.deepcopy(target_img)
    source_keypoints = target_keypoints

    pts = source_keypoints.copy()

    face_edge_start_index = 2

    pts[INDEX_FACE_OVAL[face_edge_start_index:-face_edge_start_index], 1] = target_mask_keypoints[face_edge_start_index:-face_edge_start_index, 1]

    # pts = pts[INDEX_FACE_OVAL[face_edge_start_index:-face_edge_start_index] + INDEX_NOSE_EDGE[::-1], :2]
    pts = pts[FACE_MASK_INDEX + INDEX_NOSE_EDGE[::-1], :2]

    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_img, [pts], color=(0, 0, 0))
    source_face_egde = draw_face_feature_maps(source_keypoints, mode=mode, im_edges=target_img,
                                              mouth_width = mouth_width * (256/(crop_coords[2] - crop_coords[0])), mouth_height = mouth_height * (256/(crop_coords[2] - crop_coords[0])))
    source_img = np.concatenate([source_img, source_face_egde], axis=2)
    return source_img,target_img,crop_coords

def generate_ref(img, keypoints, is_train=False, alpha = None, beta = None):
    crop_coords = crop_face(keypoints, size=img.shape[:2], is_train=is_train)
    ref_keypoints = get_image(keypoints, crop_coords, input_type='mediapipe')
    ref_img = get_image(img, crop_coords, input_type='img')

    if beta is not None:
        if alpha:
            ref_img[:, :, :3] = cv2.add(ref_img[:, :, :3], beta)
        else:
            ref_img[:, :, :3] = cv2.subtract(ref_img[:, :, :3], beta)
    ref_face_edge = draw_face_feature_maps(ref_keypoints, mode=["mouth", "nose", "eye", "oval_all","muscle"])
    ref_img = np.concatenate([ref_img, ref_face_edge], axis=2)
    return ref_img

def select_ref_index(driven_keypoints, n_ref = 5, ratio = 1/3.):
    # 根据嘴巴开合程度，选取开合最大的那一半
    lips_distance = np.linalg.norm(
        driven_keypoints[:, INDEX_LIPS_INNER[5]] - driven_keypoints[:, INDEX_LIPS_INNER[-5]], axis=1)
    selected_index_list = np.argsort(lips_distance).tolist()[int(len(lips_distance) * ratio):]
    ref_img_index_list = random.sample(selected_index_list, n_ref)  # 从当前视频选n_ref个图片
    return ref_img_index_list

def get_ref_images_fromVideo(cap, ref_img_index_list, ref_keypoints):
    ref_img_list = []
    for index in ref_img_index_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # 设置要获取的帧号
        ret, frame = cap.read()
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
        self.driven_mask_keypoints = dict_info["driven_mask_keypoints"]
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
        ref_img_list = []
        for ref_img_index in ref_img_index_list:
            ref_img = cv2.imread(self.driven_images[video_index][ref_img_index])
            # ref_img = cv2.convertScaleAbs(ref_img, alpha=self.alpha, beta=self.beta)


            ref_keypoints = self.driven_keypoints[video_index][ref_img_index]
            ref_img = generate_ref(ref_img, ref_keypoints, self.is_train, self.alpha, self.beta)

            ref_img_list.append(ref_img)
        self.ref_img = np.concatenate(ref_img_list, axis=2)

    def __getitem__(self, index):

        # 调整亮度和对比度
        # self.alpha = random.uniform(0.8,1.25)  # 缩放因子
        # self.beta = random.uniform(-50,50)  # 移位因子
        # adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        self.alpha = (random.random() > 0.5)  # 正负因子
        self.beta = np.ones([256,256,3]) * np.random.rand(3) * 20  # 色彩调整0-20个色差
        self.beta = self.beta.astype(np.uint8)


        if self.is_train:
            video_index = random.randint(0, len(self.driven_images) - 1)
            current_clip = random.randint(0, self.clip_count_list[video_index] - 1)
            ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref = self.n_ref)      # 从当前视频选n_ref个图片
            self.get_ref_images(video_index, ref_img_index_list)
        else:
            video_index = 0
            current_clip = index

            if index == 0:
                ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref=self.n_ref5)  # 从当前视频选n_ref个图片
                self.get_ref_images(video_index, ref_img_index_list)

        # target图片
        target_img = cv2.imread(self.driven_images[video_index][current_clip])
        # target_img = cv2.convertScaleAbs(target_img, alpha=self.alpha, beta=self.beta)

        target_keypoints = self.driving_keypoints[video_index][current_clip]
        target_mask_keypoints = self.driven_mask_keypoints[video_index][current_clip]

        mouth_rect = self.driving_keypoints[video_index][:, INDEX_LIPS].max(axis=1) - self.driving_keypoints[video_index][:, INDEX_LIPS].min(axis=1)
        mouth_width = mouth_rect[:, 0].max()
        mouth_height = mouth_rect[:, 1].max()

        # source_img, target_img,crop_coords = generate_input(target_img, target_keypoints, target_mask_keypoints, self.is_train)
        source_img, target_img,crop_coords = generate_input(target_img, target_keypoints, target_mask_keypoints, self.is_train, mode=["mouth_bias", "nose", "eye"],
                                                            mouth_width = mouth_width, mouth_height = mouth_height)

        target_img = target_img/255.
        source_img = source_img/255.
        ref_img = self.ref_img / 255.

        # tensor
        source_tensor = torch.from_numpy(source_img).float().permute(2, 0, 1)
        ref_tensor = torch.from_numpy(ref_img).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_img).float().permute(2, 0, 1)
        return source_tensor, ref_tensor, target_tensor

    def __len__(self):
        if self.is_train:
            return len(self.driven_images)
        else:
            return len(self.driven_images[0])
        # return self.sample_num
def data_preparation(train_video_list):
    img_all = []
    keypoints_all = []
    mask_all = []
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 0 、4、8
    for i in tqdm.tqdm(train_video_list):
        # for i in ["xiaochangzhang/00004"]:
        model_name = i
        img_filelist = glob.glob("{}/image/*.png".format(model_name))
        img_filelist.sort()
        if len(img_filelist) == 0:
            continue
        img_all.append(img_filelist)

        Path_output_pkl = "{}/keypoint_rotate.pkl".format(model_name)
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)
        keypoints_all.append(images_info[:, main_keypoints_index, :2])

        Path_output_pkl = "{}/face_mat_mask.pkl".format(model_name)
        with open(Path_output_pkl, "rb") as f:
            mat_list, face_pts_mean_personal = pickle.load(f)

        face_pts_mean_personal = face_pts_mean_personal[INDEX_FACE_OVAL]
        face_mask_pts = np.zeros([len(mat_list), len(face_pts_mean_personal), 2])
        for index_ in range(len(mat_list)):
            # img = np.zeros([1000,1000,3], dtype=np.uint8)
            # img = cv2.imread(img_filelist[index_])

            rotationMatrix = mat_list[index_]

            keypoints = np.ones([4, len(face_pts_mean_personal)])
            keypoints[:3, :] = face_pts_mean_personal.T
            driving_mask = rotationMatrix.dot(keypoints).T
            face_mask_pts[index_] = driving_mask[:, :2]



            # for coor in driving_mask:
            #     # coor = (coor +1 )/2.
            #     cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
            # cv2.imshow("a", img)
            # cv2.waitKey(30)
        mask_all.append(face_mask_pts)

    print("train size: ", len(img_all))
    dict_info = {}
    dict_info["driven_images"] = img_all
    dict_info["driven_keypoints"] = keypoints_all
    dict_info["driving_keypoints"] = keypoints_all
    dict_info["driven_mask_keypoints"] = mask_all
    return dict_info


def generate_input_pixels(img, keypoints, rotationMatrix, pixels_mouth, mask_keypoints, coords_array):
    # 根据关键点决定正方形裁剪区域
    crop_coords = crop_face(keypoints, size=img.shape[:2], is_train=False)
    target_keypoints = get_image(keypoints[:, :2], crop_coords, input_type='mediapipe')

    # 画出嘴部像素图
    pixels_mouth_coords = rotationMatrix.dot(coords_array).T
    pixels_mouth_coords = pixels_mouth_coords[:, :2].astype(int)
    pixels_mouth_coords = (pixels_mouth_coords[:, 1], pixels_mouth_coords[:, 0])

    source_face_egde = np.zeros_like(img, dtype=np.uint8)
    # out_frame = img.copy()
    frame = pixels_mouth.reshape(15, 30, 3).clip(0, 255).astype(np.uint8)

    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (150, 100))
    sharpen_image = frame.astype(np.float32)
    mean_ = int(np.mean(sharpen_image))
    max_, min_ = mean_ + 60, mean_ - 60
    sharpen_image = (sharpen_image - min_) / (max_ - min_) * 255.
    sharpen_image = sharpen_image.clip(0, 255).astype(np.uint8)

    sharpen_image = np.concatenate(
        [sharpen_image[:, :, np.newaxis], sharpen_image[:, :, np.newaxis], sharpen_image[:, :, np.newaxis]], axis=2)
    # sharpen_image = cv2.resize(sharpen_image, (150, 100))
    source_face_egde[pixels_mouth_coords] = sharpen_image.reshape(-1, 3)
    # cv2.imshow("sharpen_image", source_face_egde)
    # cv2.waitKey(40)

    source_face_egde = get_image(source_face_egde, crop_coords, input_type='image')
    source_face_egde = draw_face_feature_maps(target_keypoints, mode = ["nose", "eye"], im_edges=source_face_egde)
    # cv2.imshow("sharpen_image", source_face_egde)
    # cv2.waitKey(40)


    target_img = get_image(img, crop_coords, input_type='img')
    target_mask_keypoints = get_image(mask_keypoints[:, :2], crop_coords, input_type='mediapipe')
    # source_img信息：扣出嘴部区域
    source_img = copy.deepcopy(target_img)
    source_keypoints = target_keypoints
    pts = source_keypoints.copy()
    face_edge_start_index = 3
    pts[INDEX_FACE_OVAL[face_edge_start_index:-face_edge_start_index], 1] = target_mask_keypoints[
                                                                            face_edge_start_index:-face_edge_start_index,
                                                                            1]
    pts = pts[INDEX_FACE_OVAL[face_edge_start_index:-face_edge_start_index] + INDEX_NOSE_EDGE[::-1], :2]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_img, [pts], color=(0, 0, 0))
    source_img = np.concatenate([source_img, source_face_egde], axis=2)
    return source_img, target_img, crop_coords