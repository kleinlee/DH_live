import numpy as np
import cv2
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_standard_image_(img, kps, crop_coords, resize = (256, 256)):
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    (x_min, y_min, x_max, y_max) = [int(ii) for ii in crop_coords]
    new_w = x_max - x_min
    new_h = y_max - y_min
    img_new = np.zeros([new_h, new_w, c], dtype=np.uint8)

    # 确定裁剪区域上边top和左边left坐标
    top = int(y_min)
    left = int(x_min)
    # 裁剪区域与原图的重合区域
    top_coincidence = int(max(top, 0))
    bottom_coincidence = int(min(y_max, h))
    left_coincidence = int(max(left, 0))
    right_coincidence = int(min(x_max, w))
    img_new[top_coincidence - top:bottom_coincidence - top, left_coincidence - left:right_coincidence - left, :] = img[
                                                                                                                   top_coincidence:bottom_coincidence,
                                                                                                                   left_coincidence:right_coincidence,
                                                                                                                   :]

    img_new = cv2.resize(img_new, resize)
    kps = kps - np.array([left, top, 0])

    factor = resize[0]/new_w
    kps = kps * factor
    return img_new, kps

def get_standard_image(img_rgba, source_pts, source_crop_rect, out_size):
    '''
    将输入的RGBA图像和关键点点集转换为标准图像和标准顶点集。

    参数:
    img_rgba (numpy.ndarray): 输入的RGBA图像，形状为 (H, W, 4)。
    source_pts (numpy.ndarray): 源点集，形状为 (N, 3)，其中N是点的数量，每个点有三个坐标 (x, y, z)。
    source_crop_rect (tuple): 源图像的裁剪矩形，格式为 (x, y, width, height)。
    out_size (int): 输出图像的大小，格式为 (width, height)。

    返回:
    standard_img (numpy.ndarray): 标准化的图像，形状为 (out_size, out_size, 4)。
    standard_v (numpy.ndarray): 标准化的顶点集，形状为 (N, 3)。
    standard_vt (numpy.ndarray): 标准化的顶点集的纹理坐标，形状为 (N, 2)。
    '''
    source_pts[:, 2] = source_pts[:, 2] - np.max(source_pts[:, 2])
    standard_img, standard_v = get_standard_image_(img_rgba, source_pts, source_crop_rect, resize=out_size)

    standard_vt = standard_v.copy()
    standard_vt = standard_vt[:,:2]/ out_size
    return standard_img, standard_v, standard_vt

def crop_face_from_several_images(pts_array_origin, img_w, img_h):
    x_min, y_min, x_max, y_max = np.min(pts_array_origin[:, :, 0]), np.min(
        pts_array_origin[:, :, 1]), np.max(
        pts_array_origin[:, :, 0]), np.max(pts_array_origin[:, :, 1])
    new_w = (x_max - x_min) * 2
    new_h = (y_max - y_min) * 2
    center_x = (x_max + x_min) / 2.
    center_y = y_min + (y_max - y_min) * 0.25
    x_min, y_min, x_max, y_max = int(center_x - new_w / 2), int(center_y - new_h / 2), int(
        center_x + new_w / 2), int(center_y + new_h / 2)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(x_max, img_w)
    y_max = min(y_max, img_h)
    new_size = min((x_max + x_min) / 2., (y_max + y_min) / 2.)
    center_x = (x_max + x_min) / 2.
    center_y = (y_max + y_min) / 2.
    x_min, y_min, x_max, y_max = int(center_x - new_size), int(center_y - new_size), int(
        center_x + new_size), int(center_y + new_size)
    return np.array([x_min, y_min, x_max, y_max])

def crop_face_from_image(kps, crop_rotio = [0.6,0.6,0.65,1.35]):
    '''
    只为了裁剪图片
    :param kps:
    :param crop_rotio:
    :param standard_size:
    :return:
    '''
    x2d = kps[:, 0]
    y2d = kps[:, 1]
    w_span = x2d.max() - x2d.min()
    h_span = y2d.max() - y2d.min()
    crop_size = int(2*max(h_span, w_span))
    center_x = (x2d.max() + x2d.min()) / 2.
    center_y = (y2d.max() + y2d.min()) / 2.
    # 确定裁剪区域上边top和左边left坐标，中心点是(x2d.max() + x2d.min()/2, y2d.max() + y2d.min()/2)
    y_min = int(center_y - crop_size*crop_rotio[2])
    y_max = int(center_y + crop_size*crop_rotio[3])
    x_min = int(center_x - crop_size*crop_rotio[0])
    x_max = int(center_x + crop_size*crop_rotio[1])
    return np.array([x_min, y_min, x_max, y_max])

def check_keypoint(img, pts_):
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 0 、4、8
    for coor in pts_:
        # coor = (coor +1 )/2.
        cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
    cv2.imshow("a", img)
    cv2.waitKey(-1)

if __name__ == "__main__":
    from talkingface.mediapipe_utils import detect_face_mesh
    import glob

    image_list = glob.glob(r"F:\C\AI\CV\TalkingFace\OpenGLRender_0830\face_rgba/*.png")
    image_list.sort()
    for index, img_path in enumerate(image_list):
        img_primer_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        source_pts = detect_face_mesh([img_primer_rgba[:, :, :3]])[0]
        img_primer_rgba = cv2.cvtColor(img_primer_rgba, cv2.COLOR_BGRA2RGBA)

        source_crop_rect = crop_face_from_image(source_pts, crop_rotio=[0.75, 0.75, 0.65, 1.35])
        standard_img, standard_v, standard_vt = get_standard_image(img_primer_rgba, source_pts, source_crop_rect,
                                                                 out_size=(750, 1000))
        print(np.max(standard_vt[:, 0]))
        print(np.max(standard_vt[:, 1]))
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        thickness = 4  # 0 、4、8
        pts_ = standard_v
        img = standard_img
        for coor in pts_:
            # coor = (coor +1 )/2.
            cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
        cv2.imshow("a", img)
        cv2.waitKey(-1)