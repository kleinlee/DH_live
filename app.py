import os.path
import shutil
import gradio as gr
import subprocess
import uuid
from data_preparation_mini import data_preparation_mini
from data_preparation_web import data_preparation_web


# 自定义 CSS 样式
css = """
#video-output video {
    max-width: 300px;
    max-height: 300px;
    display: block;
    margin: 0 auto;
}
"""

video_dir_path = ""
# 假设你已经有了这两个函数
def data_preparation(video1, resize_option):
    global video_dir_path
    # 处理视频的逻辑
    video_dir_path = "video_data/{}".format(uuid.uuid4())
    data_preparation_mini(video1, video_dir_path, resize_option)
    data_preparation_web(video_dir_path)

    return "视频处理完成，保存至目录{}".format(video_dir_path)

def demo_mini(audio):
    global video_dir_path
    # 生成视频的逻辑
    audio_path = audio  # 解包元组
    wav_path = "video_data/tmp.wav"
    ffmpeg_cmd = "ffmpeg -i {} -ac 1 -ar 16000 -y {}".format(audio_path, wav_path)
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)
    output_video_name = "video_data/tmp.mp4"
    asset_path = os.path.join(video_dir_path, "assets")
    from demo_mini import interface_mini
    interface_mini(asset_path, wav_path, output_video_name)
    return output_video_name  # 返回生成的视频文件路径

# 启动网页的函数
def launch_server():
    global video_dir_path
    asset_path = os.path.join(video_dir_path, "assets")
    target_path = os.path.join("web_demo", "static", "assets")

    # 如果目标目录存在，先删除
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    # 将 asset_path 目录下的所有文件拷贝到 web_demo/static/assets 目录下
    shutil.copytree(asset_path, target_path)

    # 启动 server.py
    subprocess.Popen(["python", "web_demo/server.py"])

    return "访问 http://localhost:8888/static/MiniLive_new.html"

# 定义 Gradio 界面
def create_interface():
    with gr.Blocks(css=css) as demo:
        # 标题
        gr.Markdown("# 视频处理与生成工具")

        # 第一部分：上传静默视频和说话视频
        gr.Markdown("## 第一部分：视频处理")
        gr.Markdown("""
        - **静默视频**：时长建议在 5-30 秒之间，嘴巴不要动(保持闭嘴或微张)。嘴巴如果有动作会影响效果，请认真对待。
        """)
        with gr.Row():
            with gr.Column():
                video1 = gr.Video(label="上传静默视频", elem_id="video-output", sources="upload")
        # 增加可选项
        resize_option = gr.Checkbox(label="是否转为最高720P（适配手机）", value=True)
        process_button = gr.Button("处理视频")
        process_output = gr.Textbox(label="处理结果")

        # 分隔线
        gr.Markdown("---")

        # 第二部分：上传音频文件并生成视频
        gr.Markdown("## 第二部分：测试语音生成视频(不支持linux和MacOS，请跳过此步)")
        gr.Markdown("""
        - 上传音频文件后，点击“生成视频”按钮，程序会调用 `demo_mini` 函数完成推理并生成视频。
        - 此步骤用于初步验证结果。网页demo请执行第三步。
        """)
        # audio = gr.Audio(label="上传音频文件")

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="上传音频文件", type="filepath")
                generate_button = gr.Button("生成视频")
            with gr.Column():
                video_output = gr.Video(label="生成的视频", elem_id="video-output")

        # 分隔线
        gr.Markdown("---")

        # 第三部分：启动网页
        gr.Markdown("## 第三部分：启动网页")
        launch_button = gr.Button("启动网页")
        gr.Markdown("""
        - **注意**：本项目使用了 WebCodecs API，该 API 仅在安全上下文（HTTPS 或 localhost）中可用。因此，在部署或测试时，请确保您的网页在 HTTPS 环境下运行，或者使用 localhost 进行本地测试。
                """)
        launch_output = gr.Textbox(label="启动结果")
        gr.Markdown("""
        - 点击“启动网页”按钮后，会启动 `server.py`，提供一个模拟对话服务。
        - 在 `static/js/dialog.js` 文件中，找到第 1 行，将 server_url=`http://localhost:8888/eb_stream` 替换为您自己的对话服务网址。例如：
          ```bash
          https://your-dialogue-service.com/eb_stream
          ```
        - `server.py` 提供了一个模拟对话服务的示例。它接收 JSON 格式的输入，并流式返回 JSON 格式的响应。
        # API 接口说明

## 输入 JSON 格式

| 字段名       | 必填 | 类型   | 说明                                                                 | 默认值 |
|--------------|------|--------|----------------------------------------------------------------------|--------|
| `input_mode` | 是   | 字符串 | 输入模式，可选值为 `"text"` 或 `"audio"`，分别对应文字对话和语音对话输入 | "audio"     |
| `prompt`     | 条件 | 字符串 | 当 `input_mode` 为 `"text"` 时必填，表示用户输入的对话内容           | 无     |
| `audio`      | 条件 | 字符串 | 当 `input_mode` 为 `"audio"` 时必填，表示 Base64 编码的音频数据      | 无     |
| `voice_speed`| 否   | 字符串 | TTS 语速，可选                                                     | ""     |
| `voice_id`   | 否   | 字符串 | TTS 音色，可选                                                     | ""     |

## 输出 JSON 格式（流式返回）

| 字段名     | 必填 | 类型   | 说明                                                                 | 默认值   |
|------------|------|--------|----------------------------------------------------------------------|----------|
| `text`     | 是   | 字符串 | 返回的部分对话文本                                                  | 无       |
| `audio`    | 否   | 字符串 | Base64 编码的音频数据，可选                                         | 无       |
| `endpoint` | 是   | 布尔   | 是否为对话的最后一个片段，`true` 表示结束                           | `false`  |

---

#### 输入输出示例
```json
{
    "input_mode": "text",
    "prompt": "你好，今天天气怎么样？",
    "voice_speed": "",
    "voice_id": ""
}
输出
{
    "text": "今天天气晴朗，温度适宜。",
    "audio": "SGVsbG8sIFdvcm...",
    "endpoint": false
}
```
        """)
        # 第四部分：商业授权和更新
        gr.Markdown("## 第四部分：完整服务与更新")
        gr.Markdown("""
                - 可访问www.matesx.com 体验完整服务。
                - 商业授权（去除logo）：访问www.matesx.com/authorized.html, 上传你生成的combined_data.json.gz, 授权后下载得到新的combined_data.json.gz，覆盖原文件即可去除logo。
                - 人物切换：已开放功能，可自己整改，官方后续会完善。
                - 未来12个月会持续更新效果，可以关注公众号”Mates数字生命“获取即时动态。
                """)


        # 绑定按钮点击事件
        process_button.click(data_preparation, inputs=[video1, resize_option], outputs=process_output)
        generate_button.click(demo_mini, inputs=audio, outputs=video_output)
        launch_button.click(launch_server, outputs=launch_output)

    return demo

# 创建 Gradio 界面并启动
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()