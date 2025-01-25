# 项目说明

## 目录结构

本项目目录结构如下：
```bash
项目根目录/
├── static/                  # 静态资源文件夹
│   ├── assets/              # 人物形象资源文件夹
│   ├── common/              # 公共资源文件夹
│   ├── css/                 # CSS样式文件夹
│   ├── js/                  # JavaScript脚本文件夹
│   ├── DHLiveMini.wasm      # AI推理组件
│   ├── dialog.html          # 对话页面
│   └── MiniLive.html        # 数字人视频流主页面
└── server.py                # 启动网页服务的Python程序
```
### 运行项目

启动服务：
python server.py
打开浏览器，访问 http://localhost:8888/MiniLive.html

## 配置说明

### 1. 替换对话服务网址

在 static/js/dialog.js 文件中，找到第35行，将 http://localhost:8888/eb_stream 替换为您自己的对话服务网址。例如：
https://your-dialogue-service.com/eb_stream

### 2. 模拟对话服务

server.py 提供了一个模拟对话服务的示例。它接收JSON格式的输入，并流式返回JSON格式的响应。示例代码如下：

输入 JSON：
```bash
{
    "prompt": "用户输入的对话内容"
}
```
输出 JSON（流式返回）：
```bash
{
    "text": "返回的部分对话文本",
    "audio": "base64编码的音频数据",
    "endpoint": false  // 是否为对话的最后一个片段，true表示结束
}
```
### 3. 更换人物形象

要更换人物形象，请将新形象包中的文件替换 assets 文件夹中的对应文件。确保新文件的命名和路径与原有文件一致，以避免引用错误。

### 4. WebCodecs API 使用注意事项

本项目使用了 WebCodecs API，该 API 仅在安全上下文（HTTPS 或 localhost）中可用。因此，在部署或测试时，请确保您的网页在 HTTPS 环境下运行，或者使用 localhost 进行本地测试。

