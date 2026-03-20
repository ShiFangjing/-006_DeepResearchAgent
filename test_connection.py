# api key:43e3531f791743bebb9ef087296753f27efb883b1e7e4da0aaf69484d8bf21fa

# import requests
#
# url = "http://10.120.72.45:8000/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": "Bearer 43e3531f791743bebb9ef087296753f27efb883b1e7e4da0aaf69484d8bf21fa"
# }
# data = {
#     "model": "Qwen3-VL-32B",
#     "messages": [{"role": "user", "content": "Hello"}],
# }
#
# response = requests.post(url, headers=headers, json=data)
# print(response.json())

import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    api_key="sk-Arr9UYr2DlFA5sVznZ49B2uOdHJvUdBfnnvpccwfUQrNKFOd",
    base_url="https://api.fe8.cn/v1"
)


async def chat_with_ai():
    """异步聊天函数 - 使用 create 方法"""
    try:
        # v1.x 版本直接使用 create，但使用 AsyncOpenAI 客户端
        chat_completion = await async_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "讲个笑话",
                }
            ],
            model="claude-3-opus-20240229",
        )

        response_content = chat_completion.choices[0].message.content
        print("AI回复:", response_content)
        return response_content

    except Exception as e:
        print(f"请求出错: {e}")
        return None


# 运行
asyncio.run(chat_with_ai())