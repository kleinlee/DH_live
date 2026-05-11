# import numpy as np
# import cv2
# import os
#
# face_fusion_mask = np.zeros([128, 128], dtype = np.uint8)
# for i in range(17):
#     face_fusion_mask[i:-i,i:-i] = min(255, 16*i)
#
# cv2.imwrite("face_fusion_mask.png", face_fusion_mask)


# # from mini_live.obj.wrap_utils import index_wrap
# # image2 = cv2.imread("bs_texture.png")
# # image3 = np.zeros([12, 256, 3], dtype=np.uint8)
# # image3[:, :len(index_wrap)] = image2[:, index_wrap]
# # cv2.imwrite("bs_texture_halfFace.png", image3)
#
#
from PIL import Image, ImageDraw
from talkingface.models.DINet_mini import model_size, input_height,input_width
# 1. 构造一个全黑的100*100的图片
image = Image.new('RGB', (100, 100), color=(0, 0, 0))
draw = ImageDraw.Draw(image)

# 2. 在图片中构造19个矩形
for i in range(19):
    size = 98 - 2 * i
    color = (14 * i, 14 * i, 14 * i)
    x0 = (100 - size) // 2
    y0 = (100 - size) // 2
    x1 = x0 + size
    y1 = y0 + size
    draw.rounded_rectangle([x0, y0, x1, y1], radius=25, fill=color)

image.save('final_image.png')
image.show()

# 3. 图片按照高度分为20、60、20三个区域
region1 = image.crop((0, 0, 100, 20))
region2 = image.crop((0, 20, 100, 80))
region3 = image.crop((0, 80, 100, 100))

# 对第一个、第三个区域resize为100*8的区域
region1_resized = region1.resize((100, 7))
region3_resized = region3.resize((100, 7))

# 将三个区域concatenate起来形成新图片
new_image1 = Image.new('RGB', (100, 74))
new_image1.paste(region1_resized, (0, 0))
new_image1.paste(region2, (0, 7))
new_image1.paste(region3_resized, (0, 67))

# 4. 新图片按照宽度分为20、60、20三个区域
region1_width = new_image1.crop((0, 0, 20, 74))
region2_width = new_image1.crop((20, 0, 80, 74))
region3_width = new_image1.crop((80, 0, 100, 74))

# 对第一个、第三个区域resize为8*76的区域
region1_width_resized = region1_width.resize((7, 74))
region3_width_resized = region3_width.resize((7, 74))

# 将三个区域concatenate起来，再次形成新图片
new_image2 = Image.new('RGB', (74, 74))
new_image2.paste(region1_width_resized, (0, 0))
new_image2.paste(region2_width, (7, 0))
new_image2.paste(region3_width_resized, (67, 0))

# 5. 新图片resize为（input_width, input_height）
final_image = new_image2.resize((input_width, input_height))

# 保存最终图片
final_image.save('mouth_fusion_mask.png')
final_image.show()