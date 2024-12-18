import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
class RenderModel:
    def __init__(self, window_size):
        self.window_size = window_size
        if not glfw.init():
            raise Exception("glfw can not be initialized!")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        print(window_size[0], window_size[1])
        self.window = glfw.create_window(window_size[0], window_size[1], "Face Render window", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")
        glfw.set_window_pos(self.window, 100, 100)
        glfw.make_context_current(self.window)
        # shader 设置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.program = compileProgram(compileShader(open(os.path.join(current_dir, "shader/prompt.vsh")).readlines(), GL_VERTEX_SHADER),
                                       compileShader(open(os.path.join(current_dir, "shader/prompt.fsh")).readlines(), GL_FRAGMENT_SHADER))
        self.VBO = glGenBuffers(1)
        self.render_verts = None
        self.render_face = None
        self.face_pts_mean = None

    def setContent(self, vertices_, face):
        glfw.make_context_current(self.window)
        self.render_verts = vertices_
        self.render_face = face
        glUseProgram(self.program)
        # set up vertex array object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.GenVBO(vertices_)
        self.GenEBO(face)

        # unbind VAO
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def GenEBO(self, face):
        self.indices = np.array(face, dtype=np.uint32)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def GenTexture(self, img, texture_index = GL_TEXTURE0):
        glfw.make_context_current(self.window)
        glActiveTexture(texture_index)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image_height, image_width = img.shape[:2]
        if len(img.shape) == 2:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, image_width, image_height, 0, GL_RED, GL_UNSIGNED_BYTE,
                         img.tobytes())
        elif img.shape[2] == 3:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
        elif img.shape[2] == 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
        else:
            print("Image Format not supported")
            exit(-1)

    # def ChangeTexture(self, img, texture_index):
    #     glBindTexture(GL_TEXTURE_2D, texture1)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #     # image = cv2.imread(os.path.join(current_dir, "face_mask2.png"), cv2.IMREAD_UNCHANGED)
    #     image = ref_image
    #     img_data = image.tobytes()
    #     image_height, image_width = image.shape[:2]
    #     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    def GenVBO(self, vertices_):
        glfw.make_context_current(self.window)
        vertices = np.array(vertices_, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 13, ctypes.c_void_p(0))
        # 顶点纹理属性
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vertices.itemsize * 13, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vertices.itemsize * 13, ctypes.c_void_p(28))

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 13, ctypes.c_void_p(44))

    def render2cv(self, out_size = (1000, 1000), mat_world=None, bs_array=None):
        glfw.make_context_current(self.window)
        # 设置正交投影矩阵
        # left = 0
        # right = standard_size
        # bottom = 0
        # top = standard_size
        # near = standard_size  # 近裁剪面距离
        # far = -standard_size  # 远裁剪面距离
        left = 0
        right = out_size[0]
        bottom = 0
        top = out_size[1]
        near = 1000  # 近裁剪面距离
        far = -1000  # 远裁剪面距离

        ortho_matrix = glm.ortho(left, right, bottom, top, near, far)
        # print("ortho_matrix: ", ortho_matrix)

        glUseProgram(self.program)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        # glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)  # 剔除背面
        glFrontFace(GL_CW)  # 通常顶点顺序是顺时针
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # # 设置视口
        # glViewport(100, 0, self.window_size[0], self.window_size[1])



        glUniform1i(glGetUniformLocation(self.program, "texture_face"), 0)
        glUniform1i(glGetUniformLocation(self.program, "texture_bs"), 1)
        glUniformMatrix4fv(glGetUniformLocation(self.program, "gWorld0"), 1, GL_FALSE, mat_world)
        glUniform1fv(glGetUniformLocation(self.program, "bsVec"), 12, bs_array.astype(np.float32))

        glUniformMatrix4fv(glGetUniformLocation(self.program, "gProjection"), 1, GL_FALSE, glm.value_ptr(ortho_matrix))
        # bind VAO
        glBindVertexArray(self.vao)
        # draw
        glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_INT, None)
        # unbind VAO
        glBindVertexArray(0)

        glfw.swap_buffers(self.window)
        glReadBuffer(GL_FRONT)
        # 从缓冲区中的读出的数据是字节数组
        data = glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGBA, GL_UNSIGNED_BYTE, outputType=None)
        rgb = data.reshape(self.window_size[1], self.window_size[0], -1).astype(np.uint8)
        return rgb

def create_render_model(out_size = (384, 384), floor = 5):
    renderModel = RenderModel(out_size)

    image2 = cv2.imread(os.path.join(current_dir, "bs_texture.png"))
    renderModel.GenTexture(image2, GL_TEXTURE1)

    from obj.obj_utils import generateRenderInfo
    render_verts, render_face = generateRenderInfo(floor = floor)
    renderModel.setContent(render_verts, render_face)
    renderModel.face_pts_mean = render_verts[:478, :3].copy()
    return renderModel

def create_render_model2(out_size = (384, 384), floor = 5):
    renderModel = RenderModel(out_size)

    image2 = cv2.imread(os.path.join(current_dir, "bs_texture.png"))
    renderModel.GenTexture(image2, GL_TEXTURE1)

    from obj.obj_utils import generateRenderInfo
    render_verts, render_face = generateRenderInfo(floor = floor)
    renderModel.setContent(render_verts, render_face)
    renderModel.face_pts_mean = render_verts[:478, :3].copy()
    return renderModel

model_size_ = 384

# 示例使用
if __name__ == "__main__":
    import pickle
    import cv2
    import time
    import numpy as np
    import glob
    import random
    from OpenGL.GL import *
    import os

    start_time = time.time()
    standard_size = 384
    crop_rotio = [0.5, 0.5, 0.5, 0.5]
    out_w = int(standard_size*(crop_rotio[0] + crop_rotio[1]))
    out_h = int(standard_size*(crop_rotio[2] + crop_rotio[3]))
    out_size = (out_w, out_h)
    renderModel = create_render_model((int(out_w*0.8), int(out_h*0.8)), floor = 20)

    from talkingface.mediapipe_utils import detect_face_mesh
    from obj.image_utils import crop_face_from_image, get_standard_image
    from obj.obj_utils import faceModeling, NewFaceVerts

    image_list = glob.glob(r"F:\C\AI\CV\TalkingFace\OpenGLRender_0830/face_rgba/*.png")
    image_list.sort()
    from talkingface.utils import crop_mouth,main_keypoints_index
    for index, img_path in enumerate(image_list):
        img_primer_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        source_pts = detect_face_mesh([img_primer_rgba[:, :, :3]])[0]
        img_primer_rgba = cv2.cvtColor(img_primer_rgba, cv2.COLOR_BGRA2RGBA)
        source_crop_rect = crop_mouth(source_pts[main_keypoints_index],img_primer_rgba.shape[1], img_primer_rgba.shape[0])
        standard_img, standard_v, standard_vt = get_standard_image(img_primer_rgba, source_pts, source_crop_rect,
                                                                   out_size=(
                                                                   int(standard_size * (crop_rotio[0] + crop_rotio[1])),
                                                                   int(standard_size * (
                                                                               crop_rotio[2] + crop_rotio[3]))))

        source_crop_pts = standard_v

        # from obj.image_utils import check_keypoint
        # check_keypoint(standard_img, source_crop_pts)

        render_verts = renderModel.render_verts.copy()
        render_verts[:len(source_crop_pts), :3] = source_crop_pts
        w = standard_size * (crop_rotio[0] + crop_rotio[1])
        h = standard_size * (crop_rotio[2] + crop_rotio[3])
        render_verts[:len(source_crop_pts), 3] = source_crop_pts[:, 0] / w
        render_verts[:len(source_crop_pts), 4] = source_crop_pts[:, 1] / h
        render_verts,Rmat = NewFaceVerts(render_verts, source_crop_pts, renderModel.face_pts_mean)

        image2 = standard_img.copy()
        renderModel.GenTexture(image2, GL_TEXTURE0)
        renderModel.GenVBO(render_verts)

        rotate_max = 20
        for i in range(-rotate_max * 5, rotate_max * 5, 5):
            # bs = np.array([i*1, 0, 0, 0,0,0], dtype=np.float32)*1.2
            bs = np.zeros([12], dtype=np.float32)
            bs[0] = i
            # rotationMatrix = mat_list__.T
            bs[1] = 1.2 * bs[1] + 5
            i = (bs[0] + bs[1]) / 3. + 3
            rgba = renderModel.render2cv(out_size=out_size, mat_world=Rmat, bs_array=bs)
            final_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imshow('scene', final_img[:, :, :3].astype(np.uint8))
            cv2.waitKey(30)
