import torch
from talkingface.models.audio2bs_lstm import Audio2Feature
import time
import random
import numpy as np
import cv2
device = "cpu"

model = Audio2Feature()
model.eval()
x = torch.ones((1, 2, 80))
h0 = torch.zeros(2, 1, 192)
c0 = torch.zeros(2, 1, 192)
y, hn, cn = model(x, h0, c0)
start_time = time.time()

from thop import profile
from thop import clever_format
flops, params = profile(model.to(device), inputs=(x, h0, c0))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)
