index_wrap = [0, 2, 11, 12, 13, 14, 15, 16, 17, 18, 32, 36, 37, 38, 39, 40, 41, 42, 43, 50, 57, 58, 61,
              62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95,
              96, 97, 98, 100, 101, 106, 116, 117, 118, 119, 123, 129, 132, 135, 136, 137, 138, 140,
              142, 146, 147, 148, 149, 150, 152, 164, 165, 167, 169, 170, 171, 172, 175, 176, 177,
              178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 191, 192, 194, 199, 200, 201, 202,
              203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 227, 234, 262, 266,
              267, 268, 269, 270, 271, 272, 273, 280, 287, 288, 291, 292, 302, 303, 304, 306, 307,
              308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
              326, 327, 329, 330, 335, 345, 346, 347, 348, 352, 358, 361, 364, 365, 366, 367, 369,
              371, 375, 376, 377, 378, 379, 391, 393, 394, 395, 396, 397, 400, 401, 402, 403, 404,
              405, 406, 407, 408, 409, 410, 411, 415, 416, 418, 421, 422, 423, 424, 425, 426, 427,
              428, 430, 431, 432, 433, 434, 435, 436, 447, 454]

# INDEX_MP_LIPS = [
# 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
# 146, 91, 181, 84, 17, 314, 405, 321, 375,
# 306, 408, 304, 303, 302, 11, 72, 73, 74, 184, 76,
# 77, 90, 180, 85, 16, 315, 404, 320, 307,
# 292, 407, 272, 271, 268, 12, 38, 41, 42, 183, 62,
# 96, 89, 179, 86, 15, 316, 403, 319, 325,
# 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
# 95, 88, 178, 87, 14, 317, 402, 318, 324,
# ]
INDEX_MP_LIPS_LOWER = [
146, 91, 181, 84, 17, 314, 405, 321, 375,
77, 90, 180, 85, 16, 315, 404, 320, 307,
96, 89, 179, 86, 15, 316, 403, 319, 325,
95, 88, 178, 87, 14, 317, 402, 318, 324,
]
INDEX_MP_LIPS_UPPER = [
291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
306, 408, 304, 303, 302, 11, 72, 73, 74, 184, 76,
292, 407, 272, 271, 268, 12, 38, 41, 42, 183, 62,
308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]
index_lips_upper_wrap = []
for i in INDEX_MP_LIPS_UPPER:
    for j in range(len(index_wrap)):
        if index_wrap[j] == i:
            index_lips_upper_wrap.append(j)
index_lips_lower_wrap = []
for i in INDEX_MP_LIPS_LOWER:
    for j in range(len(index_wrap)):
        if index_wrap[j] == i:
            index_lips_lower_wrap.append(j)
print(index_lips_upper_wrap[:11] + index_lips_upper_wrap[33:44][::-1])
print(index_lips_lower_wrap[:9] + index_lips_upper_wrap[27:36][::-1])
exit(-1)
if __name__ == "__main__":
    # index_edge_wrap = [111,43,57,21,76,59,68,67,78,66,69,168,177,169,170,161,176,123,159,145,208]
    # index_edge_wrap = [110,60,79,108,61,58,73,74,62,75,77,175,164,174,173,160,163,205,178,162,207]
    index_edge_wrap = [110, 60, 79, 108, 61, 58, 73, 67, 78, 66, 69, 168, 177, 169, 173, 160, 163, 205, 178, 162, 207]
    index_edge_wrap_upper = [111, 110, 51, 52, 53, 54, 48, 63, 56, 47, 46, 1, 148, 149, 158, 165, 150, 156, 155, 154,
                             153, 207, 208]
    import numpy as np


    def readObjFile(filepath):
        v_ = []
        face = []
        with open(filepath) as f:
            # with open(r"face3D.obj") as f:
            content = f.readlines()
        for i in content:
            if i[:2] == "v ":
                v0, v1, v2 = i[2:-1].split(" ")
                v_.append(float(v0))
                v_.append(float(v1))
                v_.append(float(v2))
            if i[:2] == "f ":
                tmp = i[2:-1].split(" ")
                for ii in tmp:
                    a = ii.split("/")[0]
                    a = int(a) - 1
                    face.append(a)
        return v_, face


    verts_wrap, faces_wrap = readObjFile(r"wrap.obj")
    verts_wrap = np.array(verts_wrap).reshape(-1, 3)
    vert_mid = verts_wrap[index_edge_wrap[:4] + index_edge_wrap[-4:]].mean(axis=0)

    face_verts_num = len(verts_wrap)
    index_new_edge = []
    new_vert_list = []
    for i in range(len(index_edge_wrap)):
        index = index_edge_wrap[i]
        new_vert = verts_wrap[index] + (verts_wrap[index] - vert_mid) * 0.3
        new_vert[2] = verts_wrap[index, 2]
        new_vert_list.append(new_vert)
        index_new_edge.append(len(index_wrap) + i)
    for i in range(len(index_edge_wrap) - 1):
        faces_wrap.extend([index_edge_wrap[i], face_verts_num + i, index_edge_wrap[(i + 1) % len(index_edge_wrap)]])
        faces_wrap.extend([index_edge_wrap[(i + 1) % len(index_edge_wrap)], face_verts_num + i,
                           face_verts_num + (i + 1) % len(index_edge_wrap)])

    verts_wrap = np.concatenate([verts_wrap, np.array(new_vert_list).reshape(-1, 3)], axis=0)

    v_teeth, face_teeth = readObjFile("modified_teeth_upper.obj")
    v_teeth2, face_teeth2 = readObjFile("modified_teeth_lower.obj")

    faces_wrap = faces_wrap + [i + len(verts_wrap) for i in face_teeth] + [i + len(verts_wrap) + len(v_teeth) // 3 for i
                                                                           in
                                                                           face_teeth2]

    verts_wrap = np.concatenate([verts_wrap, np.array(v_teeth).reshape(-1, 3)], axis=0)
    verts_wrap = np.concatenate([verts_wrap, np.array(v_teeth2).reshape(-1, 3)], axis=0)

    # 边缘-1 正常0 上嘴唇2 下嘴唇2.01 上牙3 下牙4
    verts_wrap2 = np.zeros([len(verts_wrap), 5])
    verts_wrap2[:, :3] = verts_wrap
    verts_wrap2[index_lips_upper_wrap, 3] = 2.0
    verts_wrap2[index_lips_lower_wrap, 3] = 2.01
    verts_wrap2[-36:-18, 3] = 3
    verts_wrap2[-18:, 3] = 4
    verts_wrap2[index_edge_wrap_upper, 3] = -1
    verts_wrap2[index_new_edge, 3] = -1
    verts_wrap2[:, 4] = range(len(verts_wrap2))

    with open("face_wrap_entity.obj", "w") as f:
        for i in verts_wrap2:
            f.write("v {:.3f} {:.3f} {:.3f} {:.02f} {:.0f}\n".format(i[0], i[1], i[2], i[3], i[4]))
        for i in range(len(faces_wrap) // 3):
            f.write(
                "f {0} {1} {2}\n".format(faces_wrap[3 * i] + 1, faces_wrap[3 * i + 1] + 1, faces_wrap[3 * i + 2] + 1))

    # f 240 247 254
    # f 240 254 255

    # f 233 231 250
    # f 231 264 250