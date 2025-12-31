from unsloth import FastLanguageModel
import torch

# 重新加载刚才微调好的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llama3_financial_analyst_checkpoint", # 这里填你刚才保存的路径
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 准备一条最新的新闻 (你可以去 Google 搜一条今天的 NVDA 新闻填进去)
# 比如：昨天 NVDA 跌了，新闻说是因为反垄断调查
news_headline = "NVIDIA has been granted permission to sell H200 chips to China, with 25% of the sales revenue going to the US government"
ticker = "NVDA"
date = "2025-12-09"

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the following financial news headline. Predict the stock movement (UP/DOWN) for the next trading day and provide a brief reasoning.

### Input:
Ticker: {}
Date: {}
Headline: {}

### Response:
"""

inputs = tokenizer(
    [alpaca_prompt.format(ticker, date, news_headline)], 
    return_tensors = "pt"
).to("cuda")

print("🤖 AI 分析师正在思考...")
outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
result = tokenizer.batch_decode(outputs)[0]

# 提取 Response 后的内容
print("\n" + "="*30)
print(result.split("### Response:")[-1].strip())
print("="*30)