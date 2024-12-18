import numpy as np
import cv2
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

INDEX_FACE_EDGE = [
    234, 127, 162, 21,
    54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
    389, 356,
    454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150,
    136, 172, 58, 132,
    93,
]
def readObjFile(filepath):
    with_vn = False
    with_vt = False
    v_ = []
    vt = []
    vn = []
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
        if i[:3] == "vt ":
            with_vt = True
            vt0,vt1 = i[3:-1].split(" ")
            vt.append(float(vt0))
            vt.append(float(vt1))
        if i[:3] == "vn ":
            with_vn = True
            vn0,vn1,vn2 = i[3:-1].split(" ")
            vn.append(float(vn0))
            vn.append(float(vn1))
            vn.append(float(vn2))
        if i[:2] == "f ":
            tmp = i[2:-1].split(" ")
            for ii in tmp:
                a = ii.split("/")[0]
                a = int(a) - 1
                face.append(a)
    if not with_vn:
        vn = [0 for i in v_]
    if not with_vt:
        vt = [0 for i in range(len(v_)//3*2)]
    return v_, vt, vn, face

def generateRenderInfo_mediapipe():
    v_face, vt_face, vn_face, face_face = readObjFile(os.path.join(current_dir,"../obj/obj_mediapipe/face3D.obj"))
    v_teeth, vt_teeth, vn_teeth, face_teeth = readObjFile(os.path.join(current_dir,"../obj/obj_mediapipe/modified_teeth_upper.obj"))
    v_teeth2, vt_teeth2, vn_teeth2, face_teeth2 = readObjFile(os.path.join(current_dir,"../obj/obj_mediapipe/modified_teeth_lower.obj"))

    v_, vt, vn, face = (
        v_face + v_teeth + v_teeth2, vt_face + vt_teeth + vt_teeth2, vn_face + vn_teeth + vn_teeth2,
        face_face + [i + len(v_face)//3 for i in face_teeth] + [i + len(v_face)//3 + len(v_teeth)//3 for i in face_teeth2])
    v_ = np.array(v_).reshape(-1, 3)

    # v_[:, 1] = -v_[:, 1]

    # 0-2: verts   3-4: vt  5:category 6: index 7-10 bone_weight 11-12 another vt
    vertices = np.zeros([len(v_), 13])
    # vertices = np.zeros([len(pts_array_), 6])

    vertices[:, :3] = v_
    vertices[:, 3:5] = np.array(vt).reshape(-1, 2)
    vertices[:, 11:13] = np.array(vt).reshape(-1, 2)
    vertices[:, 12] = 1 - vertices[:, 12]
    # vertices[:, 5] = 0
    # 脸部为0，眼睛1，上牙2，下牙3
    vertices[468:478, 5] = 1.
    vertices[478:478 + 18, 5] = 2.
    vertices[478 + 18:478 + 36, 5] = 3.
    vertices[:, 6] = list(range(len(v_)))
    return vertices, face

def generateRenderInfo(floor = 5):
    v_face, vt_face, vn_face, face_face = readObjFile(os.path.join(current_dir,"../obj/obj_mediapipe/face3D.obj"))
    v_teeth, vt_teeth, vn_teeth, face_teeth = readObjFile(os.path.join(current_dir,"../obj/obj_mediapipe/modified_teeth_upper.obj"))
    v_teeth2, vt_teeth2, vn_teeth2, face_teeth2 = readObjFile(os.path.join(current_dir,"../obj/obj_mediapipe/modified_teeth_lower.obj"))
    print(len(v_face), len(vt_face), len(vn_face), len(face_face))
    print(len(v_teeth)//3, len(vt_teeth), len(vn_teeth), len(face_teeth))
    print(len(v_face)//3 + len(v_teeth)//3 + len(v_teeth2)//3)

    v_, vt, vn, face = (
        v_face + v_teeth + v_teeth2, vt_face + vt_teeth + vt_teeth2, vn_face + vn_teeth + vn_teeth2,
        face_face + [i + len(v_face)//3 for i in face_teeth] + [i + len(v_face)//3 + len(v_teeth)//3 for i in face_teeth2])
    v_ = np.array(v_).reshape(-1, 3)

    # v_[:, 1] = -v_[:, 1]

    vertices = np.zeros([len(v_), 13])
    # vertices = np.zeros([len(pts_array_), 6])

    vertices[:, :3] = v_
    vertices[:, 3:5] = np.array(vt).reshape(-1, 2)

    # 脸部为0，眼睛1，上牙2，下牙3， 补充的为9
    vertices[468:478, 5] = 1.

    vertices[len(v_face)//3:len(v_face)//3 + len(v_teeth)//3, 5] = 2.
    vertices[len(v_face)//3 + len(v_teeth)//3:len(v_face)//3 + len(v_teeth)//3 + len(v_teeth2)//3, 5] = 3.
    vertices[:, 6] = list(range(len(v_)))
    return vertices, face


def generateWrapModel():
    v_ = []
    face = []
    filepath = os.path.join(current_dir,"../obj/obj_mediapipe/face_wrap_entity.obj")
    with open(filepath) as f:
        # with open(r"face3D.obj") as f:
        content = f.readlines()
        for i in content:
            if i[:2] == "v ":
                v0, v1, v2, v3, v4 = i[2:-1].split(" ")
                v_.append(float(v0))
                v_.append(float(v1))
                v_.append(float(v2))
                v_.append(float(v3))
                v_.append(float(v4))
            if i[:2] == "f ":
                tmp = i[2:-1].split(" ")
                for ii in tmp:
                    a = ii.split("/")[0]
                    a = int(a) - 1
                    face.append(a)
    return np.array(v_).reshape(-1, 5), face

def NewFaceVerts(render_verts, source_crop_pts, face_pts_mean):
    from talkingface.run_utils import calc_face_mat
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(source_crop_pts[np.newaxis, :478, :],
                                                               face_pts_mean)



    # print(face_pts_mean_personal_primer.shape)
    mat_list__ = mat_list[0].T
    # mat_list__ = np.linalg.inv(mat_list[0])
    render_verts[:478,:3] = face_pts_mean_personal_primer

    # 牙齿部分校正
    from talkingface.utils import INDEX_LIPS,main_keypoints_index,INDEX_LIPS_UPPER,INDEX_LIPS_LOWER
    # # 上嘴唇中点
    # mid_upper_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS],axis = 0)
    # mid_upper_teeth = np.mean(render_verts[478:478 + 36,:3], axis=0)
    # tmp = mid_upper_teeth - mid_upper_mouth
    # print(tmp)
    # render_verts[478:478 + 36, :2] = render_verts[478:478 + 36, :2] - tmp[:2]

    # 上嘴唇中点
    mid_upper_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS_UPPER],axis = 0)
    mid_upper_teeth = np.mean(render_verts[478:478 + 18,:3], axis=0)
    tmp = mid_upper_teeth - mid_upper_mouth
    print(tmp)
    render_verts[478:478 + 18, :2] = render_verts[478:478 + 18, :2] - tmp[:2]

    # 下嘴唇中点
    mid_lower_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS_LOWER],axis = 0)
    mid_lower_teeth = np.mean(render_verts[478:478 + 18,:3], axis=0)
    tmp = mid_lower_teeth - mid_lower_mouth
    print(tmp)
    render_verts[478 + 18:478 + 36, :2] = render_verts[478 + 18:478 + 36, :2] - tmp[:2]


    return render_verts, mat_list__
