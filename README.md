# Real-time Live Streaming Digital Human
# 实时直播数字人  [bilibili video](https://www.bilibili.com/video/BV1Ppv1eEEgj/?vd_source=53601feee498369e726af7dbc2dae349)
### Video Example


https://github.com/user-attachments/assets/7e0b5bc2-067b-4048-9f88-961afed12478


## Overview
This project is a real-time live streaming digital human powered by few-shot learning. It is designed to run smoothly on all 30 and 40 series graphics cards, ensuring a seamless and interactive live streaming experience.

### Key Features
- **Real-time Performance**: The digital human can interact in real-time with 25+ fps for common NVIDIA 30 and 40 series GPUs
- **Few-shot Learning**: The system is capable of learning from a few examples to generate realistic responses.
## Usage

### Unzip the Model File
First, navigate to the `checkpoint` directory and unzip the model file:
```bash
cd checkpoint
gzip -d -c render.pth.gz.001 > render.pth
```
### Prepare Your Video
Next, prepare your video using the data_preparation script. Replace YOUR_VIDEO_PATH with the path to your video:
```bash
python data_preparation YOUR_VIDEO_PATH
```
The result (video_info) will be stored in the ./video_data directory.
### Run with Audio File
Run the demo script with an audio file. Make sure the audio file is in .wav format with a sample rate of 16kHz and 16-bit single channel. Replace video_data/test with the path to your video_info file, video_data/audio0.wav with the path to your audio file, and 1.mp4 with the desired output video path:
```bash
python demo.py video_data/test video_data/audio0.wav 1.mp4
```
### Real-Time Run with Microphone
For real-time operation using a microphone, simply run the following command:
```bash
python demo_avatar.py
```

## Acknowledgements 
We would like to thank the contributors of [wavlip](https://github.com/Rudrabha/Wav2Lip), [dinet](https://github.com/MRzzm/DINet), [livespeechportrait](https://github.com/YuanxunLu/LiveSpeechPortraits) repositories, for their open research and contributions.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, please contact us at [kleinlee1@outlook.com].
