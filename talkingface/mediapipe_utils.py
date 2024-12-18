import numpy as np
import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def detect_face_mesh(frames):
    pts_3d = np.zeros([len(frames), 478, 3])
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for frame_index, frame in enumerate(frames):
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                image_height, image_width = frame.shape[:2]
                for face_landmarks in results.multi_face_landmarks:
                    for index_, i in enumerate(face_landmarks.landmark):
                        x_px = i.x * image_width
                        y_px = i.y * image_height
                        z_px = i.z * image_width
                        pts_3d[frame_index, index_] = np.array([x_px, y_px, z_px])
            else:
                break
    return pts_3d
def detect_face(frames):
    rect_2d = np.zeros([len(frames), 4])
    # 剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80的
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        for frame_index, frame in enumerate(frames):
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.detections or len(results.detections) > 1:
                break
            rect = results.detections[0].location_data.relative_bounding_box
            rect_2d[frame_index] = np.array([rect.xmin, rect.xmin + rect.width, rect.ymin, rect.ymin + rect.height])
    return rect_2d