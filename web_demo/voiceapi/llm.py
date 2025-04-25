from openai import OpenAI
# 豆包
base_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = ""
model_name = "doubao-pro-32k-character-241215"

# # DeepSeek
# base_url = "https://api.deepseek.com"
# api_key = ""
# model_name = "deepseek-chat"

assert api_key, "您必须配置自己的LLM API秘钥"

llm_client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)


def llm_stream(prompt):
    stream = llm_client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model=model_name,
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": prompt},
        ],
        # 响应内容是否流式返回
        stream=True,
    )
    return stream
