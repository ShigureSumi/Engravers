import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_fomc_transcripts(start_year=2020, end_year=2024):
    """
    爬取美联储FOMC新闻发布会实录（含Q&A环节）
    """
    transcript_data = []
    base_url = "https://www.federalreserve.gov"
    
    for year in range(start_year, end_year + 1):
        # 实录归档页面URL
        transcript_url = f"{base_url}/monetarypolicy/fomcpresconf{year}.htm"
        try:
            res = requests.get(transcript_url, headers=headers)
            if res.status_code != 200:
                print(f"⚠️ 无法访问{year}年实录页面")
                continue
            
            soup = BeautifulSoup(res.content, 'html.parser')
            # 查找所有实录链接
            links = soup.find_all('a')
            for link in links:
                link_text = link.get_text(strip=True)
                link_href = link.get('href', '')
                # 筛选发布会实录链接（含"Transcript"关键词）
                if 'Transcript' in link_text and 'HTML' in link_text:
                    full_href = base_url + link_href if not link_href.startswith('http') else link_href
                    # 提取日期（从链接文本或href中解析，示例："September 20, 2023 Transcript"）
                    date_str = ' '.join(link_text.split()[:-1])  # 去掉末尾的"Transcript"
                    try:
                        # 进入实录页面提取完整文本
                        trans_res = requests.get(full_href, headers=headers)
                        trans_soup = BeautifulSoup(trans_res.content, 'html.parser')
                        # 提取所有正文段落
                        paragraphs = trans_soup.find_all('p')
                        full_transcript = " ".join([p.get_text().strip() for p in paragraphs])
                        if len(full_transcript) > 1000:  # 过滤无效文本
                            transcript_data.append({
                                'Raw_Date': date_str,
                                'Year': year,
                                'Transcript_Url': full_href,
                                'Full_Transcript_Text': full_transcript
                            })
                            print(f"✅ 成功爬取{year}年{date_str}实录")
                        time.sleep(1)
                    except Exception as e:
                        print(f"❌ 爬取{full_href}失败：{e}")
        except Exception as e:
            print(f"❌ 处理{year}年实录失败：{e}")
    
    return pd.DataFrame(transcript_data)

# 执行爬取
df_transcripts = get_fomc_transcripts(2020, 2024)
df_transcripts.to_csv('fomc_transcripts_raw.csv', index=False)

import re
import pandas as pd

def extract_qa_from_transcript(full_transcript):
    """
    从完整实录中提取Q&A环节文本
    核心逻辑：1.  识别开场白结束标识；2.  筛选Q/A标识内容
    """
    if pd.isna(full_transcript):
        return ""
    
    # 方法1： 按"Opening Remarks"分割（开场白之后即为Q&A）
    qa_text = ""
    if "Opening Remarks" in full_transcript:
        # 分割为“开场白”和“Q&A”两部分，取后半部分
        qa_text = full_transcript.split("Opening Remarks")[-1]
    elif "QUESTIONS AND ANSWERS" in full_transcript:
        # 方法2： 按明确的Q&A标题分割
        qa_text = full_transcript.split("QUESTIONS AND ANSWERS")[-1]
    else:
        # 方法3： 正则匹配Q/A标识（如"Q:" "A:" "Question:" "Answer:"）
        qa_pattern = r'(Q:|A:|Question:|Answer:).*?'
        qa_matches = re.findall(qa_pattern, full_transcript, re.DOTALL)
        if qa_matches:
            qa_text = " ".join(qa_matches)
    
    # 清理Q&A文本中的噪声
    qa_text = re.sub(r'\s+', ' ', qa_text).strip()
    return qa_text

# 加载爬取的实录数据
df_transcripts = pd.read_csv('fomc_transcripts_raw.csv')
# 提取Q&A文本
df_transcripts['Transcript_Text'] = df_transcripts['Full_Transcript_Text'].apply(extract_qa_from_transcript)
# 合并到你的核心数据集（按日期匹配）
df_core = pd.read_csv('fed_project_data.csv')
# 先标准化日期（需确保两边日期格式一致，可复用第一个代码的parse_fomc_date函数）
df_transcripts['Date'] = df_transcripts.apply(lambda x: parse_fomc_date(x), axis=1)
df_core['Date'] = pd.to_datetime(df_core['Date'])
# 合并Q&A文本到核心数据集
df_merged = pd.merge(df_core, df_transcripts[['Date', 'Transcript_Text']], on='Date', how='left')
# 保存最终数据
df_merged.to_csv('fed_project_data_with_qa.csv', index=False)
print("✅ Q&A环节提取完成，已合并到核心数据集！")