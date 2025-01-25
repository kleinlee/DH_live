import json
import requests
import asyncio
import re
import base64
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="web_demo/static"), name="static")

def get_audio(text_cache, voice_speed, voice_id):
    # 读取一个语音文件模拟语音合成的结果
    with open("web_demo/static/common/test.wav", "rb") as audio_file:
        audio_value = audio_file.read()
    base64_string = base64.b64encode(audio_value).decode('utf-8')
    return base64_string

def llm_answer(prompt):
    # 模拟大模型的回答
    answer = "我会重复三遍来模仿大模型的回答，我会重复三遍来模仿大模型的回答，我会重复三遍来模仿大模型的回答。"
    return answer

def split_sentence(sentence, min_length=10):
    # 定义包括小括号在内的主要标点符号
    punctuations = r'[。？！；…，、()（）]'
    # 使用正则表达式切分句子，保留标点符号
    parts = re.split(f'({punctuations})', sentence)
    parts = [p for p in parts if p]  # 移除空字符串
    sentences = []
    current = ''
    for part in parts:
        if current:
            # 如果当前片段加上新片段长度超过最小长度，则将当前片段添加到结果中
            if len(current) + len(part) >= min_length:
                sentences.append(current + part)
                current = ''
            else:
                current += part
        else:
            current = part
    # 将剩余的片段添加到结果中
    if len(current) >= 2:
        sentences.append(current)
    return sentences

async def gen_stream(prompt, voice_speed=None, voice_id=None):
    # 访问对话大模型
    text_cache = llm_answer(prompt)
    # 对长文本进行切分，切分后的短文本依次进行语音生成并发送，以此保证流式快速响应
    sentences = split_sentence(text_cache)

    for index_, sub_text in enumerate(sentences):
        base64_string = get_audio(sub_text, voice_speed, voice_id)
        # 生成 JSON 格式的数据块
        chunk = {
            "text": sub_text,
            "audio": base64_string,
            "endpoint": index_ == len(sentences)-1
        }
        yield f"{json.dumps(chunk)}\n"  # 使用换行符分隔 JSON 块
        await asyncio.sleep(0.2)  # 模拟异步延迟

@app.post("/eb_stream")    # 前端调用的path
async def eb_stream(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    voice_speed = body.get("voice_speed")
    voice_id = body.get("voice_id")
    voice_speed = voice_speed if voice_speed != "" else None
    voice_id = voice_id if voice_id != "" else None
    return StreamingResponse(gen_stream(prompt, voice_speed, voice_id), media_type="application/json")

# 启动Uvicorn服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
