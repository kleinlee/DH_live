

<div align="center">
<img src="preview/web.jpg" width="240" />

# DH_live (mini)

> 网页 · 实时 · 移动端 · 全网最小

[中文版](README.md) | [English](README_en.md)

在线应用：[matesx.com](matesx.com)
</div>

Notes：目前项目主要维护DH_live_mini, 目前最快的2D视频数字人方案，没有之一，项目含有网页推理的案例，不依赖任何GPU，可在任何手机设备实时运行。

我们已开源超轻量级多端数字人对话引擎[MatesX](https://github.com/kleinlee/MatesX)。是DH_live进化版本。记忆 · 表情 · 动作 · 多端 · 轻量, 适配 Windows/macOS/iOS/Android/小程序
<div align="center">
<img src="preview/matesx.jpg" width="360" />
</div>
DHLive_mini手机浏览器直接推理演示 [bilibili video](https://www.bilibili.com/video/BV1UgFFeKEpp)

# 新闻
- 2025-01-26 最小化简化网页资源包，gzip资源小于2MB。简化视频数据，数据大小减半
- 2025-02-09 增加ASR入口、增加一键切换形象。
- 2025-02-27 优化渲染、去除参照视频，目前只需要一段视频即可生成。
- 2025-03-11 增加DH_live_mini的CPU支持。
- 2025-04-09 增加对IOS17以上的长视频支持。
- 2025-04-25 增加完整的实时对话服务，包含vad-asr-llm-tts-数字人全流程，请见web_demo/server_realtime.py。
- 2025-09-23 超轻量级多端数字人对话引擎[MatesX](https://github.com/kleinlee/MatesX)已开源。是DH_live进化版本。记忆 · 表情 · 动作 · 多端 · 轻量, 适配 Windows/macOS/iOS/Android/小程序

## 数字人方案对比

| 方案名称 | 单帧算力（Mflops） | 使用方式 | 脸部分辨率 | 适用设备 |
|------------------------------|-------------------|------------|------------|------------------------------------|
| Ultralight-Digital-Human（mobile） | 1100 | 单人训练 | 160 | 中高端手机APP |
| DH_live_mini | 39 | 无须训练 | 128 | 所有设备，网页&APP&小程序 |
| DH_live | 55046 | 无须训练 | 256 | 30系以上显卡 |
| duix.ai | 1200 | 单人训练 | 160 | 中高端手机APP |


### 主要特性
- **最低算力**: 推理一帧的算力39 Mflops，有多小？小于手机端大部分的人脸检测算法。
- **最小存储**：整个网页资源可以压缩到3MB！
- **无须训练**: 开箱即用，无需复杂的训练过程。

### 平台支持
- **windows**: 支持视频数据处理、离线视频合成、网页服务器。
- **linux&macOS**：支持视频数据处理、搭建网页服务器，不支持离线视频合成。
- **网页&小程序**：支持客户端直接打开（可搜索小程序"MatesX数字生命"，功能和网页版完全一致）。
- **App**：webview方式调用网页或重构原生应用。

| 平台 | Windows | Linux/macOS |
|---------------|---------------|-------------|
| 原始视频处理&网页资源准备 | ✅ | ✅ |
| 离线视频合成 | ✅ | ❌ |
| 构建网页服务器 | ✅ | ✅ |
| 实时对话 | ✅ | ✅ |

## 模型文件
所有模型文件已移至 [百度网盘](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)
[GoogleDrive](https://drive.google.com/drive/folders/1az5WEWOFmh0_yrF3I9DEyctMyjPolo8V?usp=sharing)


## 简易使用 (Gradio)
第一次使用或想获取完整流程请运行此Gradio。
```bash
python app.py
```

## 使用方法

### 创建环境
首先，导航到 `checkpoint` 目录并解压模型文件：
```bash
conda create -n dh_live python=3.11
conda activate dh_live
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd checkpoint
```
注意如果没有GPU可以安装CPU版本的pytorch

下载并解压模型文件。
### 准备视频
```bash
python data_preparation_mini.py video_data/000002/video.mp4 video_data/000002
python data_preparation_web.py video_data/000002
```
处理后的视频信息将存储在 ./video_data 目录中。
### 使用音频文件运行 ( linux and MacOS not supported!!! )
语音文件必须是单通道16K Hz的wav文件格式。
```bash
python demo_mini.py video_data/000002/assets video_data/audio0.wav 1.mp4
```
### 网页演示
请将新形象包中的assets文件(譬如video_data/000002/assets)替换 assets 文件夹中的对应文件
```bash
python web_demo/server.py
```
可以打开 localhost:8888/static/MiniLive.html。

如果想体验最佳的流式对话效果，请认真阅读 [web_demo/README.md](https://github.com/kleinlee/DH_live/blob/main/web_demo/README.md),内含完整的可商用工程。
## 授权
网页部分的商业应用涉及形象授权（去除logo）：访问[授权说明] (www.matesx.com/authorized.html)

上传你生成的combined_data.json.gz, 授权后下载得到新的combined_data.json.gz，覆盖原文件即可去除logo。
## 在线应用
访问[matesx网页程序](https://www.matesx.com)， 或小程序请搜索"MatesX数字生命"

## 算法构架图
<div align="center">
<img src="preview/AlgorithmArchitecture.png" width="480" />
</div>

## 许可证
MIT License

## 联系
<center>

| 加我微信群交流                                                   | QQ群聊 |
|-----------------------------------------------------------|-----------------------------------------------------------|
| <img src="preview/wechat.jpg" width="480" alt="MatesX 官方微信"/> | <img src="preview/qq.jpg" width="480" alt="MatesX 官方QQ"/> |

</center>