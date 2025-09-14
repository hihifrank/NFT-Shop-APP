import transformers
print(transformers.__version__)

from transformers import pipeline

# 初始化生成管道
generator = pipeline('text-generation', model='distilgpt2')
print("LLM載入成功")