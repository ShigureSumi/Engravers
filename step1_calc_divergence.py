import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. 加载 SBERT 模型
# 'all-MiniLM-L6-v2' 是一个速度快且效果极好的模型，专门用于计算句子相似度
print("正在加载 SBERT 模型 (首次运行会自动下载约 80MB)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_divergence(row):
    """
    计算 Statement 和 Transcript 之间的语义差异度
    """
    stmt = row['Statement_Text']
    trans = row['Transcript_Text']
    
    # 数据完整性检查
    if pd.isna(stmt) or pd.isna(trans):
        return None
    
    # 预处理：SBERT 有最大长度限制，通常我们取最重要的部分
    # 策略：Statement 很短，取全部；Transcript 很长，我们取前 2000 个字符(通常包含开场白)
    # 或者取 Q&A 部分（如果你的文本已经清洗过）
    stmt_vec = model.encode(stmt[:1000], convert_to_tensor=True)
    trans_vec = model.encode(trans[:2000], convert_to_tensor=True)
    
    # 计算余弦相似度 (Cosine Similarity)
    # 结果在 -1 到 1 之间
    cosine_sim = util.pytorch_cos_sim(stmt_vec, trans_vec).item()
    
    # 转化为差异度 (Divergence)
    # 1 表示完全相同，0 表示完全无关。我们希望 Divergence 越大代表风险越高。
    # Divergence = 1 - Similarity
    return 1 - cosine_sim

# --- 模拟运行 (你可以替换为你真实的 CSV 读取) ---
# 假设这是你整理好的数据
data = {
    'Date': ['2023-05-03', '2023-06-14'],
    'Statement_Text': [
        "The Committee decides to raise the target range for the federal funds rate to 5 to 5-1/4 percent.",
        "Holding the target range steady at this meeting allows the Committee to assess additional information."
    ],
    'Transcript_Text': [
        "We are prepared to do more if greater monetary policy restraint is warranted. Inflation is still too high.", 
        "We have covered a lot of ground, and the full effects of our tightening have yet to be felt."
    ]
}
df = pd.DataFrame(data)

print("开始计算语义差异度...")
df['Divergence_Score'] = df.apply(calculate_divergence, axis=1)

print("计算完成！结果预览：")
print(df)

# 保存结果，供 LSTM 使用
df.to_csv('nlp_features_ready.csv', index=False)