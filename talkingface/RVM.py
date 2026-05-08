import shutil

import onnxruntime as ort
import numpy as np
import cv2
import os
import uuid
from tqdm import tqdm

# Global ONNX session variable initialized as None
_sess = None
rec = None

import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import torch
from RobustVideoMatting.model import MattingNetwork
import os

def get_onnx_session():
    """
    Get the ONNX session (singleton pattern)
    """
    global _sess
    if _sess is None:
        _sess = MattingNetwork('resnet50').eval().cuda()  # 或 "resnet50"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        _sess.load_state_dict(torch.load(os.path.join(current_dir, '../checkpoint/rvm_resnet50.pth')))
    return _sess

def process_img_matting(frame_rgba, is_new_video = True):
    """
    Process a video to extract matting and save as WebM with alpha channel.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        str: Path to the output WebM file with alpha channel.
    """
    global rec
    # Get ONNX session (will initialize only once)
    sess = get_onnx_session()

    if is_new_video:
        # Initialize recurrent states (must match model's dtype)
        rec = [None] * 4

    frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
    frame_normalized = (frame_rgb / 255.0).astype(np.float32)
    src = torch.from_numpy(frame_normalized).float().permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        fgr, pha, *rec = sess(src, *rec, downsample_ratio=0.375)
    pha = pha.detach().squeeze(0).squeeze(0).cpu().float().numpy() * 255.0
    pha = pha.astype(np.uint8)

    final_rgba = np.concatenate([frame_rgb, pha[:, :, np.newaxis]], axis=2)
    return final_rgba

# # Example usage
# if __name__ == "__main__":
#     output_path = process_video_to_webm(r"C:\Users\kleinlee\Downloads/douyin_7104583518403038478_video.mp4", "344.webm")
#     print(f"Final output: {output_path}")