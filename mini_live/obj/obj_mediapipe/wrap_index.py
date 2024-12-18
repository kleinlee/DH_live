import numpy as np
import cv2
def readObjFile(filepath):
    v_ = []
    face = []
    with open(filepath) as f:
        # with open(r"face3D.obj") as f:
        content = f.readlines()
    for i in content:
        if i[:2] == "v ":
            v0,v1,v2 = i[2:-1].split(" ")
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

verts_face,_ =  readObjFile(r"face3D.obj")
verts_wrap,_ = readObjFile(r"wrap.obj")

verts_flame =  np.array(verts_face).reshape(-1, 3)
verts_mouth = np.array(verts_wrap).reshape(-1, 3)
index_mouthInFlame = []
for index in range(len(verts_mouth)):
    vert = verts_mouth[index]
    dist_list = []
    for i in verts_flame:
        dist_list.append(np.linalg.norm(i - vert))
    align_index = np.argmin(dist_list)
    index_mouthInFlame.append(align_index)
print(index_mouthInFlame)
# exit()

# from obj.utils import INDEX_FLAME_LIPS
# index_mouthInFlame = np.array(index_mouthInFlame, dtype = int)[INDEX_FLAME_LIPS]
# np.savetxt("index_mouthInFlame.txt", index_mouthInFlame)