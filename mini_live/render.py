import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap
from mini_live.obj.obj_utils import generateRenderInfo, generateWrapModel
from talkingface.utils import crop_mouth, main_keypoints_index
from talkingface.model_utils import device
current_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
import torch.nn.functional as F
class RenderModel_gl:
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
        self.program = compileProgram(compileShader(open(os.path.join(current_dir, "shader/prompt3.vsh")).readlines(), GL_VERTEX_SHADER),
                                       compileShader(open(os.path.join(current_dir, "shader/prompt3.fsh")).readlines(), GL_FRAGMENT_SHADER))
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

    def GenVBO(self, vertices_):
        glfw.make_context_current(self.window)
        vertices = np.array(vertices_, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))
        # 顶点纹理属性
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))

    def render2cv(self, vertBuffer, out_size = (1000, 1000), mat_world=None, bs_array=None):
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
        glUniformMatrix4fv(glGetUniformLocation(self.program, "gProjection"), 1, GL_FALSE, glm.value_ptr(ortho_matrix))

        # print("ortho_matrix: ", ortho_matrix)

        glUseProgram(self.program)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        # glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)  # 剔除背面
        glFrontFace(GL_CW)  # 通常顶点顺序是顺时针
        glClearColor(0.5, 0.5, 0.5, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # # 设置视口
        # glViewport(100, 0, self.window_size[0], self.window_size[1])


        glUniform1i(glGetUniformLocation(self.program, "texture_bs"), 0)
        glUniformMatrix4fv(glGetUniformLocation(self.program, "gWorld0"), 1, GL_FALSE, mat_world)
        glUniform1fv(glGetUniformLocation(self.program, "bsVec"), 12, bs_array.astype(np.float32))

        glUniform2fv(glGetUniformLocation(self.program, "vertBuffer"), 209, vertBuffer.astype(np.float32))

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
    renderModel_gl = RenderModel_gl(out_size)

    image2 = cv2.imread(os.path.join(current_dir, "bs_texture_halfFace.png"))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGBA)
    renderModel_gl.GenTexture(image2, GL_TEXTURE0)

    render_verts, render_face = generateRenderInfo()
    wrapModel_verts,wrapModel_face = generateWrapModel()

    renderModel_gl.setContent(wrapModel_verts, wrapModel_face)
    renderModel_gl.render_verts = render_verts
    renderModel_gl.render_face = render_face
    renderModel_gl.face_pts_mean = render_verts[:478, :3].copy()
    return renderModel_gl

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

    import torch
    from talkingface.model_utils import LoadAudioModel,Audio2bs
    from talkingface.data.few_shot_dataset import get_image

    Audio2FeatureModel = LoadAudioModel(r'../checkpoint\lstm/lstm_model_epoch_325.pkl')

    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("../checkpoint/DINet_mini/epoch_40.pth")

    start_time = time.time()
    standard_size = 256
    crop_rotio = [0.5, 0.5, 0.5, 0.5]
    out_w = int(standard_size*(crop_rotio[0] + crop_rotio[1]))
    out_h = int(standard_size*(crop_rotio[2] + crop_rotio[3]))
    out_size = (out_w, out_h)
    renderModel_gl = create_render_model((out_w, out_h), floor = 20)

    wrapModel,wrapModel_face = generateWrapModel()

    path = r"F:\C\AI\CV\DH008_few_shot/preparation_new"
    video_list = os.listdir(r"{}".format(path))
    print(video_list)
    for test_video in video_list[:10]:
        Path_output_pkl = "{}/{}/keypoint_rotate.pkl".format(path, test_video)
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)

        images_info = np.concatenate([images_info, images_info[::-1]], axis=0)

        video_path = "{}/{}/circle.mp4".format(path, test_video)
        cap = cv2.VideoCapture(video_path)
        vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        list_source_crop_rect = []
        list_video_img = []
        list_standard_img = []
        list_standard_v = []
        list_standard_vt = []
        for frame_index in range(min(vid_frame_count, len(images_info))):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            source_pts = images_info[frame_index]
            source_crop_rect = crop_mouth(source_pts[main_keypoints_index], vid_width, vid_height)

            standard_img = get_image(frame, source_crop_rect, input_type="image", resize = standard_size)
            standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize = standard_size)
            standard_vt = standard_v[:, :2] / standard_size

            list_video_img.append(frame)
            list_source_crop_rect.append(source_crop_rect)
            list_standard_img.append(standard_img)
            list_standard_v.append(standard_v)
            list_standard_vt.append(standard_vt)
        cap.release()

        renderModel_mini.reset_charactor(list_standard_img, np.array(list_standard_v)[:,main_keypoints_index])
        from talkingface.run_utils import calc_face_mat
        mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v), renderModel_gl.face_pts_mean)

        from mini_live.obj.utils import INDEX_MP_LIPS
        face_pts_mean_personal_primer[INDEX_MP_LIPS] = renderModel_gl.face_pts_mean[INDEX_MP_LIPS] * 0.4 + face_pts_mean_personal_primer[INDEX_MP_LIPS] * 0.6

        from mini_live.obj.wrap_utils import newWrapModel
        face_wrap_entity = newWrapModel(wrapModel, face_pts_mean_personal_primer)

        renderModel_gl.GenVBO(face_wrap_entity)

        wav2_dir = glob.glob(r"F:\C\AI\CV\DH008_few_shot\wav/*.wav")
        sss = random.randint(0, len(wav2_dir)-1)
        wavpath = wav2_dir[sss]
        bs_array = Audio2bs(wavpath, Audio2FeatureModel)[5:] * 0.5

        import uuid
        task_id = str(uuid.uuid1())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = "{}.mp4".format(task_id)

        videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))

        for frame_index in range(len(mat_list)):
            if frame_index >= len(bs_array):
                continue
            bs = np.zeros([12], dtype=np.float32)
            bs[:6] = bs_array[frame_index, :6]
            # bs[2] = frame_index* 5

            verts_frame_buffer = np.array(list_standard_vt)[frame_index, index_wrap, :2].copy() * 2 - 1

            rgba = renderModel_gl.render2cv(verts_frame_buffer, out_size=out_size, mat_world=mat_list[frame_index].T, bs_array=bs)
            # rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

            # rgba = cv2.resize(rgba, (128, 128))
            rgba = rgba[::2, ::2, :]

            gl_tensor = torch.from_numpy(rgba/255.).float().permute(2, 0, 1).unsqueeze(0)
            source_tensor = cv2.resize(list_standard_img[frame_index], (128, 128))
            source_tensor = torch.from_numpy(source_tensor/255.).float().permute(2, 0, 1).unsqueeze(0)

            warped_img = renderModel_mini.interface(source_tensor.to(device), gl_tensor.to(device))

            image_numpy = warped_img.detach().squeeze(0).cpu().float().numpy()
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            image_numpy = image_numpy.clip(0, 255)
            image_numpy = image_numpy.astype(np.uint8)
            # print(image_numpy.shape)
            # cv2.imshow('scene', image_numpy)
            # cv2.waitKey(40)
            x_min, y_min, x_max, y_max = list_source_crop_rect[frame_index]

            img_face = cv2.resize(image_numpy, (x_max - x_min, y_max - y_min))
            img_bg = list_video_img[frame_index][:,:,:3]
            img_bg[y_min:y_max, x_min:x_max, :3] = img_face[:,:,:3]
            # cv2.imshow('scene', img_bg[:,:,::-1])
            # cv2.waitKey(40)
            # print(time.time())

            videoWriter.write(img_bg[:,:,::-1])
        videoWriter.release()
        os.makedirs("output", exist_ok=True)
        val_video = "output/{}.mp4".format(task_id + "_2")

        wav_path = wavpath
        os.system(
            "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p {}".format(save_path, wav_path, val_video))
        os.remove(save_path)

    cv2.destroyAllWindows()