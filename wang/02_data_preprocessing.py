import pandas as pd
import json
import random

# ================= 配置 =================
INPUT_CSV = "my_custom_fin_dataset_2025.csv" # 上一步爬下来的文件
OUTPUT_JSONL = "llama3_finetune_data.jsonl"

def generate_reasoning(row):
    """
    根据涨跌标签和标题，生成一段'伪'分析理由 (Chain of Thought)。
    这样训练出来的模型，不仅会吐出 UP/DOWN，还能瞎编一段看似合理的解释。
    """
    ticker = row['Ticker']
    label = row['Label']
    headline = str(row['Headline'])
    
    reasons_up = [
        f"This news indicates strong market confidence in {ticker}.",
        f"Positive sentiment surrounding {ticker}'s operations is likely to drive investor demand.",
        f"The market is reacting favorably to recent developments for {ticker}.",
        "Growth prospects for the tech sector remain robust, supporting this upward movement."
    ]
    
    reasons_down = [
        f"Investors might react negatively to the uncertainty surrounding {ticker}.",
        f"This news adds to the bearish sentiment for {ticker} in the short term.",
        f"Potential headwinds mentioned in the news could pressure {ticker}'s stock price.",
        "Profit-taking and market corrections are likely following this news."
    ]
    
    # 简单的关键词匹配增强理由真实性
    if "revenue" in headline.lower() or "earnings" in headline.lower():
        if label == "UP": return f"Better-than-expected financial performance suggests {ticker} is undervalued."
        if label == "DOWN": return f"Disappointing financial metrics raise concerns about {ticker}'s growth trajectory."
        
    if "launch" in headline.lower() or "new" in headline.lower():
        if label == "UP": return f"Product innovation is a key driver for {ticker}'s future revenue streams."
    
    # 默认随机理由
    if label == "UP":
        return random.choice(reasons_up)
    elif label == "DOWN":
        return random.choice(reasons_down)
    else:
        return f"The market impact of this news for {ticker} appears neutral or mixed."

def convert_to_llama_format():
    print("🔄 正在将 CSV 转换为 Llama-3 训练格式...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ 找不到 {INPUT_CSV}，请先运行上一步的爬虫代码！")
        return

    # 过滤掉 Neutral (为了训练效果更明显，通常只训练涨跌)
    # 如果你想保留 Neutral 也可以，注释掉下面这行
    df = df[df['Label'] != "NEUTRAL"].copy()
    
    print(f"   -> 有效样本数: {len(df)}")
    
    dataset = []
    
    for idx, row in df.iterrows():
        # 1. 构建输入
        input_text = f"Ticker: {row['Ticker']}\nDate: {row['Date']}\nHeadline: {row['Headline']}"
        
        # 2. 生成理由
        reasoning = generate_reasoning(row)
        
        # 3. 构建输出 (CoT 风格)
        output_text = f"Prediction: {row['Label']}\nAnalysis: {reasoning}"
        
        # 4. 组合成 Alpaca/Llama 格式
        entry = {
            "instruction": "Analyze the following financial news headline. Predict the stock movement (UP/DOWN) for the next trading day and provide a brief reasoning.",
            "input": input_text,
            "output": output_text
        }
        dataset.append(entry)
        
    # 保存为 JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write('\n')
            
    print(f"✅ 转换完成！保存为: {OUTPUT_JSONL}")
    print("   样本预览:")
    print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    convert_to_llama_format()