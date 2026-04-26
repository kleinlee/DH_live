import numpy as np
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap,index_edge_wrap_upper
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
# print(index_lips_upper_wrap[:11] + index_lips_upper_wrap[33:44][::-1])
# print(index_lips_lower_wrap[:9] + index_lips_upper_wrap[27:36][::-1])
if __name__ == "__main__":
    def readObjFile(filepath):
        v_ = []
        face = []
        with open(filepath) as f:
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

    # 在wrap.obj基础上增加眉心的一个点
    verts_, _ = readObjFile(r"face3D.obj")
    vert_top = np.array(verts_).reshape(-1, 3)[10]

    verts_wrap = np.concatenate([verts_wrap, vert_top.reshape(-1, 3)], axis=0)
    for i in range(len(index_edge_wrap_upper) - 1):
        faces_wrap.extend([index_edge_wrap_upper[i], index_edge_wrap_upper[i + 1], 209])

    with open("wrap2.obj", "w") as f:
        for i in verts_wrap:
            f.write("v {} {} {}\n".format(i[0], i[1], i[2]))
        for i in range(len(faces_wrap)//3):
            f.write("f {} {} {}\n".format(faces_wrap[3*i + 0]+1, faces_wrap[3*i + 1]+1, faces_wrap[3*i + 2]+1))


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

    # 补全牙齿后侧的拓扑
    face_outer_vert_index = [145, 159, 123, 176, 161, 170,
                             43, 57, 21, 76, 59, 68]

    for i in range(5):
        faces_wrap.extend([face_outer_vert_index[i], face_outer_vert_index[6+i], face_outer_vert_index[i+1]])
        faces_wrap.extend([face_outer_vert_index[i + 1], face_outer_vert_index[6 + i], face_outer_vert_index[6 + i + 1]])

    # 边缘-1 正常0 上嘴唇2 下嘴唇2.01 上牙3 下牙4
    verts_wrap2 = np.zeros([len(verts_wrap), 5])
    verts_wrap2[:, :3] = verts_wrap
    verts_wrap2[index_lips_upper_wrap, 3] = 2.0
    verts_wrap2[index_lips_lower_wrap, 3] = 2.01
    verts_wrap2[-36:-18, 3] = 3
    verts_wrap2[-18:, 3] = 4
    verts_wrap2[index_edge_wrap_upper, 3] = -1
    verts_wrap2[index_new_edge, 3] = -1
    verts_wrap2[209, 3] = -1
    verts_wrap2[face_outer_vert_index, 3] = -2
    verts_wrap2[:, 4] = range(len(verts_wrap2))

    with open("face_wrap_entity.obj", "w") as f:
        for i in verts_wrap2:
            f.write("v {:.3f} {:.3f} {:.3f} {:.02f} {:.0f}\n".format(i[0], i[1], i[2], i[3], i[4]))
        for i in range(len(faces_wrap) // 3):
            f.write(
                "f {0} {1} {2}\n".format(faces_wrap[3 * i] + 1, faces_wrap[3 * i + 1] + 1, faces_wrap[3 * i + 2] + 1))
    print("Done!")