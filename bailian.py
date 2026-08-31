import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-ws-H.ELYHHHR.h28J.MEYCIQCaR_P30n4egznmCQbTNflgUpBARW3Npt3hrJOaN6z09wIhALmqPuO7JZK4Qw4hN0RQtYOWImzrDQqrilcAjYFmaPOP",
    base_url="https://ws-oy04s3orhdjo6ai8.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen3.8-max",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
)
print(completion.model_dump_json())