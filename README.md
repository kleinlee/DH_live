# Mobile and Web Real-time Live Streaming Digital Human! 
# 实时数字人 全网最快
Notes：目前项目主要维护DH_live_mini, 目前最快的数字人方案，没有之一，项目含有网页推理的案例，不依赖任何GPU，可在任何手机设备实时运行。原版DH_live已不再获支持，希望慎重考虑使用。原版使用方法参见[here](https://github.com/kleinlee/DH_live/blob/main/README_DH_live.md)。

DHLive_mini手机浏览器直接推理[bilibili video](https://www.bilibili.com/video/BV1UgFFeKEpp)

![微信图片_20250209153828](https://github.com/user-attachments/assets/32650fac-3885-4c98-886f-66258ef891a7)


# News
- 2025-01-26 最小化简化网页资源包，gzip资源小于2MB。简化视频数据，数据大小减半
- 2025-02-09 增加ASR入口、增加一键切换形象。
- 2025-02-27 优化渲染、去除参照视频，目前只需要一段视频即可生成。
- 2025-03-11 增加DH_live_mini的CPU支持。

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
  
### 平台支持
- **windows**: 支持视频数据处理、离线视频合成、网页服务器。
- **linux&macOS**：支持视频数据处理、搭建网页服务器，不支持离线视频合成。
- **网页&小程序**：支持客户端直接打开。
- **App**：webview方式调用网页或重构原生应用。


| 平台            | Windows       | Linux/macOS |
|---------------|---------------|-------------|
| 原始视频处理&网页资源准备 | ✅             | ✅           |
| 离线视频合成        | ✅             | ❌           |   
| 构建网页服务器       | ✅             | ✅         | 
| 实时对话          | ✅             | ✅           |

## Easy Usage (Gradio)
第一次使用或想获取完整流程请运行此Gradio。
```bash
python app.py
```

## Usage

### Create Environment
First, navigate to the `checkpoint` directory and unzip the model file:
```bash
conda create -n dh_live python=3.11
conda activate dh_live
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd checkpoint
```
注意如果可以选择安装CPU版本的pytorch:  pip install torch

unzip checkpoint files from [baiduNetDisk](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)
### Prepare Your Video
```bash
python data_preparation_mini.py video_data/000002/video.mp4 video_data/000002
python data_preparation_web.py video_data/000002
```
处理后的视频信息将存储在 ./video_data 目录中。
### Run with Audio File ( linux and MacOS not supported!!! )
语音文件必须是单通道16K Hz的wav文件格式。
```bash
python demo_mini.py video_data/000002/assets video_data/audio0.wav 1.mp4
```
### Web demo
请将新形象包中的assets文件(譬如video_data/000002/assets)替换 assets 文件夹中的对应文件
```bash
python web_demo/server.py
```
可以打开 localhost:8888/static/MiniLive.html。
## License
DH_live is licensed under the MIT License.

## 联系
|  加我好友，请备注“进群”，拉你进去微信交流群。| 进入QQ群聊，分享看法和最新咨询。 |
|-------------------|----------------------|
| ![微信交流群](https://github.com/user-attachments/assets/b1f24ebb-153b-44b1-b522-14f765154110) | ![QQ群聊](https://github.com/user-attachments/assets/29bfef3f-438a-4b9f-ba09-e1926d1669cb) |
