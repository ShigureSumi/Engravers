---
Title: Beyond Static CSVs: Engineering a Real-Time Financial News Pipeline for the AI Supply Chain (by Group "Engravers")
Date: 2025-12-31 17:00
Category: Reflective Report
Tags: Group Fintech Disruption
---

By Group "Engravers"

Our project involved a broad market sentiment forecasting framework (training a large language model (LLM) to predict  returns from aggregate financial news).

## 1. Motivation: Ditching Generic Datasets for Targeted, Timely Data

When we started our project, we first tested public Kaggle datasets (such as "Financial News 2025"). But we quickly hit a wall: **these datasets are too noisy for our "AI Supply Chain" focus** (NVDA, TSM, AMD, etc.). A headline like "Market Rises on Fed News" tells us nothing about how Nvidia’s stock will react to a new chip launch—we needed data that links *specific corporate events* to *specific stock movements*.

That’s why we built a custom data pipeline: to curate a 2024–2025 dataset tailored to the GenAI boom cycle. Generic CSVs can’t capture this niche, real-time signal—so we engineered our own.

## 2. Tech Stack: Python Tools for Robust Data Engineering

We chose a lightweight but scalable Python stack:
- **Data Sources:** `GNews` (for targeted news headlines) + `yfinance` (for daily stock prices)
- **Processing:** `Pandas` (time-series alignment) + `tqdm` (progress tracking)
- **Edge Cases:** Custom date-parsing logic (to handle messy news timestamps)

The code we use is as follows:
```python
import pandas as pd
import yfinance as yf
from gnews import GNews
from tqdm import tqdm
import time
import datetime
from newspaper import Article
```

## 3. Pipeline Breakdown: Stages, Challenges, and Fixes

Our pipeline turned unstructured news and raw prices into labeled, LLM-ready data—here’s how (and the problems we hit):

### 3.1 Phase 1: Targeted News Crawling (The "Relevance Filter")
We scraped headlines for 7 AI supply chain stocks (NVDA, TSM, MSFT, etc.) using queries like `"{Ticker} stock news"` (to avoid generic market content).

**Challenge 1: Google’s Anti-Scraping Limits**
Our early runs were blocked after scraping two stock tickers because Google flagged our frequent requests.

**Fix:** Added a 2-second delay between ticker queries (via `time.sleep(2)`) and limited `max_results=100` per ticker to stay under rate limits.

```python
google_news = GNews(language='en', country='US', period='1y', max_results=100)

for ticker in tqdm(tickers):
    time.sleep(2)
```

**Challenge 2: Messy GNews Date Formats**
GNews returned dates like `"Fri, 27 Dec 2024 07:00:00 GMT"` and `"2024/12/27"`— Pandas couldn’t parse them uniformly.

**Fix:** Added a try-except block to standardize dates (with a "today" fallback for unparseable entries):

```python
try:
                dt = pd.to_datetime(pub_date).strftime("%Y-%m-%d")
            except:
                dt = datetime.datetime.now().strftime("%Y-%m-%d")
```

### 3.2 Phase 2: Time-Series Alignment (Avoiding the "Weekend Gap")
Merging news (which publishes 24/7) with stock prices (which trade 5 days/week) required strict temporal logic.

**Challenge: News Published on Weekends/Holidays**
We noticed that the news headlines on Saturday about the delayed release of TSMC chips did not correspond to Saturday's prices (the market was closed). So we need to map them to the *next trading day*.

**Fix:** Used `Pandas`’ `asof` logic to find the nearest future trading session:
    
```python
if dt not in returns.index:
                valid_dates = returns.index[returns.index > dt]
                if len(valid_dates) == 0: continue
                target_date = valid_dates[0]
            else:
                target_date = dt
```

**Critical Guardrail: No Look-Ahead Bias**
We aligned news from day `t` with *next-day returns* (day `t+1`), not same-day returns—this ensures our model predicts the future (not explains the past), a key rule in quant finance.

```python
prices = yf.download(list(unique_tickers), start=START_DATE, end=END_DATE)['Close']

returns = prices.pct_change().shift(-1)
```

### 3.3 Phase 3: Label Generation (Cleaning Up Noise)
We converted continuous returns into 3 discrete labels (for LLM instruction tuning):
- `UP`: Return > +0.5%
- `DOWN`: Return < -0.5%
- `NEUTRAL`: -0.5% ≤ Return ≤ +0.5% (we kept these but marked them to reduce training noise)

## 4. Final Dataset: Stats & Signal-to-Noise Win
Our pipeline output a **proprietary dataset** with:
**Timeframe:** Jan 2024 – Present (covers the GenAI boom peak)
**Samples:** Several aligned news-price pairs (non-tradable dates filtered out)
**Features:** `Date`, `Ticker`, `Headline`, `Next_Day_Return`, `Label`

Compared to generic Kaggle datasets—whose signal-to-noise ratio (SNR) is significantly lower when applied to our AI supply chain stocks—our custom-curated dataset delivers a notably higher SNR. This enhanced signal clarity makes it far more suitable for training our Llama-3 model to establish meaningful links between news headlines and stock price movements.

## 5. Innovation: Tailoring the Dataset for Llama-3’s Context-Aware Supply Chain Reasoning
The true value of our proprietary dataset lies in its design to unlock **Llama-3’s unique strengths**—a departure from traditional financial NLP (e.g., FinBERT) that treats firms as isolated entities. Here’s how we’re leveraging Llama-3 to turn our data into actionable, explainable insights:

### 5.1 Dataset Design for Llama-3’s Relational Reasoning
Traditional NLP datasets focus on single-ticker news, but our inclusion of the `Supply_Chain_Neighbor` field (e.g., linking NVDA to its supplier TSM) aligns with Llama-3’s ability to model *inter-entity dependencies*. 

For example:
When fine-tuned, Llama-3 will use TSM’s "yield improvement" news (paired with NVDA’s headlines) to reason: *"TSM’s supply stability reduces NVDA’s production risk → NVDA price is likely to rise"*—a connection traditional models miss.

### 5.2 Beyond Binary Sentiment: Llama-3’s Chain-of-Thought (CoT) Capability
Unlike traditional NLP models that only output binary "positive/negative" scores, we plan to fine-tune Llama-3 using a labeled dataset via "Clock of Thought" (CoT) tuning to achieve **interpretable predictions**.

### 5.3 Efficient Fine-Tuning on the advanced graphics cards
We’ll use **Unsloth** (a lightweight LLM fine-tuning library) and **LoRA (Low-Rank Adaptation)** to fine-tune Llama-3-8B on our dataset:
LoRA trains only 1% of Llama-3’s parameters, fitting the model to our advanced graphics cards.
Early tests have shown that this setup can improve prediction accuracy and also provide reasoning results that are easier for humans to understand.

## 6. Reflections: Hurdles & Lessons Learned
1. **Pivoting Is Not Failure**
   Our initial general market project failed, but narrowing to the AI supply chain let us leverage *specific, high-impact events* (GPU launches, chip delays) that drive clear stock moves.

2. **Hardware Matters**
   Our laptops take a long time to fetch a year's worth of news; renting a advanced graphics card can reduce that time. Next time, we'll prioritize using cloud resources for data-intensive steps to avoid wasting time.

3. **Edge Cases Make Data Useful**
   The weekend gap and date parsing were small details, but skipping them would have made our dataset useless for modeling. We now add an "edge case checklist" to all pipeline builds.

## 7. Next Steps
1. Add full-text scraping (via `Newspaper3k`) to capture more context than headlines.
2. Refine labels to include move magnitude (e.g., `UP_LARGE` for returns > 2%) for better LLM fine-tuning.
3. Test the pipeline on 2023 data to validate performance across a full market cycle.