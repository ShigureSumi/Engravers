# 优化第二个代码的calculate_divergence函数，优先使用Q&A文本
def extract_qa_from_capiq(transcript):
    """
    从CapIQ实录中提取Q&A环节
    """
    if pd.isna(transcript):
        return ""
    # CapIQ实录的Q&A标识更规范，直接分割
    if "Questions and Answers" in transcript:
        qa_part = transcript.split("Questions and Answers")[1]
        return qa_part[:3000]  # 取Q&A前3000字符（足够包含核心问答）
    elif "Q&A" in transcript:
        qa_part = transcript.split("Q&A")[1]
        return qa_part[:3000]
    else:
        return transcript[:2000]  # 兼容无标识的情况

# 在calculate_divergence函数中调用
def calculate_divergence(row):
    stmt = row['Statement_Text']
    trans = row['Transcript_Text']
    
    if pd.isna(stmt) or pd.isna(trans):
        return None
    
    # 先提取Q&A部分
    trans_qa = extract_qa_from_capiq(trans)
    stmt_vec = model.encode(stmt[:1000], convert_to_tensor=True)
    trans_vec = model.encode(trans_qa, convert_to_tensor=True)  # 用Q&A文本计算差异度
    
    cosine_sim = util.pytorch_cos_sim(stmt_vec, trans_vec).item()
    return 1 - cosine_sim