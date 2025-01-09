# Mobile and Web Real-time Live Streaming Digital Human! 
# 实时数字人 全网最快
Notes：目前项目主要维护DH_live_mini, 目前最快的数字人方案，没有之一，项目含有网页推理的案例，不依赖任何GPU，可在任何手机设备实时运行。原版DH_live已不再或支持，希望慎重考虑使用。原版使用方法参见[here](https://github.com/kleinlee/DH_live/blob/main/README_DH_live.md)。

DHLive_mini手机浏览器直接推理[bilibili video](https://www.bilibili.com/video/BV1pWkwYWEn4)

# 数字人方案对比

| 方案名称                     | 单帧算力（Mflops） | 使用方式   | 脸部分辨率 | 适用设备                           |
|------------------------------|-------------------|------------|------------|------------------------------------|
| Ultralight-Digital-Human（mobile） | 1100              | 单人训练   | 160        | 中高端手机APP                      |
| DH_live_mini                  | 39                | 无须训练   | 128        | 所有设备，网页&APP&小程序          |
| DH_live                       | 55046            | 无须训练   | 256        | 30系以上显卡                       |
| duix.ai                      | 1200             | 单人训练   | 160        | 中高端手机APP                      |

### checkpoint
All checkpoint files are moved to [baiduNetDisk](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)

### Key Features
- **最低算力**: 推理一帧的算力39 Mflops，有多小？小于手机端大部分的人脸检测算法。
- **最小存储**：整个网页资源可以压缩到3MB！
- **无须训练**: 开箱即用，无需复杂的训练过程。
  
## Usage

### Create Environment
First, navigate to the `checkpoint` directory and unzip the model file:
```bash
conda create -n dh_live python=3.12
conda activate dh_live
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd checkpoint
```
unzip checkpoint files from [baiduNetDisk](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)
### Prepare Your Video
```bash
python data_preparation_web.py YOUR_VIDEO_PATH
```
处理后的视频信息将存储在 ./video_data 目录中。
### Run with Audio File
```bash
python demo_mini.py video_data/test video_data/audio0.wav 1.mp4
```
### Web demo
```bash
python data_preparation_web.py YOUR_VIDEO_PATH
cd web_demo
python server.py
```
可以打开 localhost:8888/static/MiniLive.html, 可以手机上打开。
## License
DH_live is licensed under the MIT License.

## 联系
| 进入QQ群聊，分享看法和最新咨询。 | 加我好友，请备注“进群”，拉你进去微信交流群。 |
|-------------------|----------------------|
| ![QQ群聊](https://github.com/user-attachments/assets/29bfef3f-438a-4b9f-ba09-e1926d1669cb) | ![微信交流群](https://github.com/user-attachments/assets/b1f24ebb-153b-44b1-b522-14f765154110) |
