import numpy as np

def ModifyObjFile(path):
    with_vn = False
    verts = []
    vt = []
    vn = []
    face = []
    map_face = {}
    with open(path) as f:
        content = f.readlines()
    for i in content:
        if i[:2] == "v ":
            verts.append(i)
        if i[:3] == "vt ":
            vt.append(i)
        if i[:3] == "vn ":
            with_vn = True
            vn.append(i)
        if i[:2] == "f ":
            tmp = i[2:-1].split(" ")
            face.extend(tmp)
    print(len(verts),len(vt),len(vn),len(face),len(set(face)))
    set_face = set(face)
    print(len(set_face))
    for index,i in enumerate(set_face):
        map_face[i] = index



    with open("modified_" + path, "w") as f:
        for i in set_face:
            index = int(i.split("/")[0]) - 1
            f.write(verts[index])
        for i in set_face:
            index = int(i.split("/")[1]) - 1
            f.write(vt[index])
        if with_vn:
            for i in set_face:
                index = int(i.split("/")[2]) - 1
                f.write(vn[index])
            for i in range(len(face) // 3):
                f.write("f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n".format(map_face[face[3 * i]] + 1,
                                                                         map_face[face[3 * i + 1]] + 1,
                                                                         map_face[face[3 * i + 2]] + 1))
        else:
            for i in range(len(face) // 3):
                f.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(map_face[face[3 * i]] + 1,
                                                                         map_face[face[3 * i + 1]] + 1,
                                                                         map_face[face[3 * i + 2]] + 1))


ModifyObjFile("teeth_lower.obj")
ModifyObjFile("teeth_upper.obj")

# ModifyObjFile("teeth_new.obj")
#
# import cv2
# import numpy as np
# img0 = cv2.imread(r"texture/Std_Lower_Teeth_diffuse.jpg")
# img1 = cv2.imread(r"texture/Std_Tongue_diffuse.jpg")
# img2 = cv2.imread(r"texture/Std_Upper_Teeth_diffuse.jpg")
# img = np.concatenate([img0, img1, img2], axis = 1)
# cv2.imwrite("teeth.png", cv2.resize(img, (256*3,256)))
#
# img0 = cv2.imread(r"texture/Std_Lower_Teeth_normal.png")
# img1 = cv2.imread(r"texture/Std_Tongue_normal.jpg")
# img2 = cv2.imread(r"texture/Std_Upper_Teeth_normal.png")
# img = np.concatenate([img0, img1, img2], axis = 1)
# cv2.imwrite("teeth_normal.png", cv2.resize(img, (256*3,256)))
# print(img[:100,:100])
# img0 = cv2.imread(r"teeth.png").astype(float)
# img1 = cv2.imread(r"teeth_ao.png", cv2.IMREAD_GRAYSCALE).astype(float)/255.
#
# img0[:,:,0] = img0[:,:,0]*img1
# img0[:,:,1] = img0[:,:,1]*img1
# img0[:,:,2] = img0[:,:,2]*img1
#
# cv2.imwrite("teeth_.png", img0.astype(np.uint8))