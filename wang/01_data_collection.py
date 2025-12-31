import pandas as pd
import yfinance as yf
from gnews import GNews
from tqdm import tqdm
import time
import datetime
from newspaper import Article

# ================= 配置区域 =================
# 我们只关注这一条“AI 黄金供应链”，逻辑最强
TARGET_TICKERS = ["NVDA", "TSM", "AMD", "MSFT", "AAPL", "GOOGL", "META"]

# 时间范围 (建议爬过去 1-2 年的，数据量正好适合微调)
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=800)).strftime("%Y-%m-%d") # 过去一年
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

OUTPUT_FILE = "my_custom_fin_dataset_2025.csv"

# ================= 1. 新闻爬虫 (Google News) =================
def fetch_google_news(tickers):
    print(f"🕷️ [1/3] 启动 Google News 爬虫 ({START_DATE} - {END_DATE})...")
    
    google_news = GNews(language='en', country='US', period='2y', max_results=100) # 这里的 period 可以根据需要调整
    all_news = []
    
    for ticker in tqdm(tickers):
        # 搜索关键词：公司股票代码 + "stock" 或者 "revenue" 等词，提高相关性
        keyword = f"{ticker} stock news"
        json_resp = google_news.get_news(keyword)
        
        for article in json_resp:
            # 简单清洗
            title = article.get('title', '')
            pub_date = article.get('published date', '')
            url = article.get('url', '')
            
            # 过滤掉太短的标题
            if len(title) < 10: continue
            
            # 格式化日期 (GNews 返回的格式很乱，需要统一)
            # 这里简化处理，尝试解析，解析不了就跳过
            try:
                # GNews 格式通常是: "Fri, 27 Dec 2024 07:00:00 GMT"
                dt = pd.to_datetime(pub_date).strftime("%Y-%m-%d")
            except:
                dt = datetime.datetime.now().strftime("%Y-%m-%d") # 兜底
                
            all_news.append({
                'Date': dt,
                'Ticker': ticker,
                'Headline': title,
                'Source': article.get('publisher', {}).get('title', 'Unknown'),
                'URL': url
            })
            
        # 礼貌延时，防止 Google 封 IP
        time.sleep(2)
        
    df_news = pd.DataFrame(all_news)
    # 去重 (不同关键词可能爬到同一条新闻)
    df_news = df_news.drop_duplicates(subset=['Headline'])
    print(f"   -> 爬取完成，共 {len(df_news)} 条原始新闻")
    return df_news

# ================= 2. 股价获取与标签生成 =================
def align_prices_and_label(news_df):
    print("📈 [2/3] 下载股价并生成【涨跌标签】...")
    
    final_data = []
    unique_tickers = news_df['Ticker'].unique()
    
    # 批量下载股价
    prices = yf.download(list(unique_tickers), start=START_DATE, end=END_DATE)['Close']
    
    # 计算 T+1 收益率
    # Shift(-1) 因为我们用【今天的新闻】预测【明天的涨跌】
    returns = prices.pct_change().shift(-1)
    
    for idx, row in tqdm(news_df.iterrows(), total=len(news_df)):
        ticker = row['Ticker']
        date_str = row['Date']
        
        try:
            # 查找当天的收益率
            # 注意：如果新闻是周末发的，我们要对应到周一的收益率
            # 这里用 asof 找最近的交易日
            dt = pd.to_datetime(date_str)
            if dt not in returns.index:
                # 找最近的一个未来交易日
                valid_dates = returns.index[returns.index > dt]
                if len(valid_dates) == 0: continue
                target_date = valid_dates[0]
            else:
                target_date = dt
                
            ret = returns.loc[target_date][ticker]
            
            # 打标签 (Labeling)
            # 涨 (UP): > 0.5%
            # 跌 (DOWN): < -0.5%
            # 震荡 (NEUTRAL): -0.5% ~ 0.5%
            
            label = "NEUTRAL"
            if ret > 0.005: label = "UP"
            elif ret < -0.005: label = "DOWN"
            
            # 只有明确的涨跌才适合训练 LLM，震荡数据可能会造成混淆
            # 策略：保留震荡数据但标记，或者直接丢弃。为了训练效果，建议保留或丢弃。
            
            # 构建 CoT (思维链) 训练数据所需的 "Reasoning" (伪造/推断)
            # 注意：在真实微调时，我们会让 LLM 自己学 reasoning，
            # 这里我们先准备好 Input (Headline) 和 Output (Label)
            
            final_data.append({
                'Date': date_str,
                'Ticker': ticker,
                'Headline': row['Headline'],
                'Source': row['Source'],
                'Next_Day_Return': ret,
                'Label': label
            })
            
        except Exception as e:
            continue
            
    df_final = pd.DataFrame(final_data)
    # 过滤掉 Neutral (可选，如果你想做二分类)
    # df_final = df_final[df_final['Label'] != "NEUTRAL"]
    
    return df_final

# ================= 3. 主程序 =================
if __name__ == "__main__":
    # 1. 爬新闻
    news_df = fetch_google_news(TARGET_TICKERS)
    
    if not news_df.empty:
        # 2. 对齐股价
        dataset = align_prices_and_label(news_df)
        
        # 3. 保存
        dataset.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ 数据集构建完成！保存为: {OUTPUT_FILE}")
        print(f"   有效样本数: {len(dataset)}")
        print("   现在你可以用这个 CSV 去喂给 Llama-3 了！")
        print("\n样本预览:")
        print(dataset[['Date', 'Ticker', 'Headline', 'Label']].head())
    else:
        print("❌ 没爬到新闻，请检查网络 (可能需要梯子连接 Google)。")