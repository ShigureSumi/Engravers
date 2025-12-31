import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np

# ==========================================
# 第一部分：自动爬取 FOMC Statements
# ==========================================

def get_fomc_statements(start_year=2020, end_year=2025):
    """
    爬取美联储利率决议声明 (Statement)
    """
    data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"🔄 开始爬取 {start_year}-{end_year} 年的 FOMC 声明...")
    
    for year in range(start_year, end_year + 1):
        # FOMC 日历页面 URL
        calendar_url = f"https://www.federalreserve.gov/monetarypolicy/fomccalendars{year}.htm"
        
        try:
            res = requests.get(calendar_url, headers=headers)
            if res.status_code != 200:
                print(f"⚠️ 无法访问 {year} 年日历")
                continue
                
            soup = BeautifulSoup(res.content, 'html.parser')
            
            # 查找所有会议 (HTML 结构通常在 'fomc-meeting' class 中)
            meetings = soup.find_all('div', class_='fomc-meeting')
            
            for meeting in meetings:
                # 提取日期 (格式通常是 "Month Day-Day")
                date_div = meeting.find('div', class_='fomc-meeting__date')
                if not date_div: continue
                date_str = date_div.get_text(strip=True)
                
                # 寻找 Statement 链接
                links = meeting.find_all('a')
                stmt_url = None
                for link in links:
                    if 'Statement' in link.get_text() and 'HTML' in link.get_text():
                        href = link.get('href')
                        if not href.startswith('http'):
                            stmt_url = "https://www.federalreserve.gov" + href
                        else:
                            stmt_url = href
                        break
                
                if stmt_url:
                    # 进入链接提取正文
                    try:
                        stmt_res = requests.get(stmt_url, headers=headers)
                        stmt_soup = BeautifulSoup(stmt_res.content, 'html.parser')
                        
                        # 提取正文：通常在 <div class="col-xs-12 col-sm-8 col-md-8"> 或 <div id="article">
                        # 我们提取所有 <p> 标签并过滤掉页脚
                        paragraphs = stmt_soup.find_all('p')
                        text_content = " ".join([p.get_text().strip() for p in paragraphs])
                        
                        # 简单清理：去掉太短的段落（通常是导航链接）
                        if len(text_content) > 500:
                            # 格式化日期: "January 28-29" -> 转换为具体日期
                            # 这里的逻辑比较复杂，我们先存原始 URL 和年份，后面统一处理日期
                            data.append({
                                'Raw_Date': date_str,
                                'Year': year,
                                'Url': stmt_url,
                                'Statement_Text': text_content
                            })
                            print(f"✅ 成功抓取: {date_str} {year}")
                            
                    except Exception as e:
                        print(f"❌ 抓取内容失败 {stmt_url}: {e}")
                    
                    time.sleep(1) # 礼貌延时
                    
        except Exception as e:
            print(f"❌ 处理年份 {year} 失败: {e}")
            
    return pd.DataFrame(data)

# ==========================================
# 第二部分：日期处理与清洗
# ==========================================

def parse_fomc_date(row):
    """
    将 'January 28-29' 这样的字符串转换为真实的 '2020-01-29'
    美联储声明通常在会议的最后一天发布
    """
    raw = row['Raw_Date']
    year = row['Year']
    
    # 提取月份和最后一天
    # 例子: "January 28-29" -> "January 29"
    # 例子: "March 15" -> "March 15"
    # 例子: "April 28-29" -> "April 29"
    # 例子: "July 31-August 1" (跨月) -> "August 1"
    
    try:
        if '-' in raw:
            # 处理跨天/跨月情况
            parts = raw.split('-')
            last_part = parts[-1].strip() # 取破折号后面部分
            
            # 如果后面部分包含月份 (e.g., "August 1")
            if any(m in last_part for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                date_str = f"{last_part} {year}"
            else:
                # 只有日期 (e.g., "29")，需要前面的月份
                first_part = parts[0].strip() # "January 28"
                month = first_part.split()[0]
                date_str = f"{month} {last_part} {year}"
        else:
            # 单日会议
            date_str = f"{raw} {year}"
            
        # 转换为 datetime 对象
        dt = datetime.strptime(date_str, "%B %d %Y")
        return dt
    except Exception as e:
        print(f"⚠️ 日期解析错误: {raw} {year} -> {e}")
        return None

# ==========================================
# 第三部分：获取 VIX 和 债券波动率数据 (yfinance)
# ==========================================

def get_market_data(start_date, end_date):
    print("🔄 正在下载 VIX (恐慌指数) 和 TLT (债券ETF) 数据...")
    
    # 1. 下载 VIX (股市波动率)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)['Close']
    vix = vix.rename("VIX_Close")
    
    # 2. 下载 TLT (20年+国债ETF) -> 计算其波动率作为 MOVE Index 的免费替代品
    # 真正的 MOVE Index (Ticker: ^MOVE) 在 Yahoo Finance 上数据经常缺失或不可用
    # 我们用 TLT 的 5日滚动标准差 来模拟债券波动
    tlt = yf.download("TLT", start=start_date, end=end_date, progress=False)['Close']
    
    # 合并
    market_df = pd.DataFrame({'VIX': vix, 'TLT_Price': tlt})
    
    # 计算未来波动率 (Target)
    # 逻辑：我们想预测 *未来5天* 的平均 VIX
    market_df['VIX_Future_5D_Avg'] = market_df['VIX'].rolling(window=5).mean().shift(-5)
    
    # 计算债券波动率代理指标 (Bond Volatility Proxy)
    # 计算 TLT 的日收益率
    market_df['TLT_Ret'] = market_df['TLT_Price'].pct_change()
    # 计算 20日 滚动波动率
    market_df['Bond_Vol_Proxy'] = market_df['TLT_Ret'].rolling(window=20).std() * np.sqrt(252)
    
    return market_df

# ==========================================
# 主程序：执行与合并
# ==========================================

# 1. 爬取文本
df_text = get_fomc_statements(2020, 2024) # 建议先跑最近几年的

# 2. 清洗日期
df_text['Date'] = df_text.apply(parse_fomc_date, axis=1)
df_text = df_text.dropna(subset=['Date']).sort_values('Date')

# 3. 获取市场数据 (范围比文本宽一点，确保有前后数据)
min_date = df_text['Date'].min() - timedelta(days=30)
max_date = df_text['Date'].max() + timedelta(days=30)
df_market = get_market_data(min_date, max_date)

# 4. 合并数据
# 我们需要把 Market Data merge 到 Text Data 上
# 注意：FOMC 声明通常在下午发布，市场反应可能在当天(收盘前)或第二天
# 这里我们匹配 "会议当天" 的数据
df_final = pd.merge_asof(
    df_text.sort_values('Date'),
    df_market.reset_index().sort_values('Date'),
    on='Date',
    direction='forward' # 如果当天非交易日，向后找最近的交易日
)

# 5. 添加 Transcript 占位符
df_final['Transcript_Text'] = ""  # 这一列留空，等待你们填入 CapIQ 数据

# 6. 保存
output_file = 'fed_project_data.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✅ 数据准备完成！")
print(f"📂 文件已保存为: {output_file}")
print(f"📊 包含列: {df_final.columns.tolist()}")
print("\n👉 接下来的步骤：")
print("1. 打开 fed_project_data.csv")
print("2. 从 Capital IQ 下载对应日期的 Press Conference Transcript")
print("3. 将 Transcript 文本粘贴到 'Transcript_Text' 列中 (或写个小脚本批量填入)")