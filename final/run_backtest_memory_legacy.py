import pandas as pd
import numpy as np
import yfinance as yf
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from datetime import timedelta
import re

# Force offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

# ================= Configuration Defaults =================
# LEGACY MODEL (News Only) - The one used in 5.1.4
DEFAULT_MODEL_PATH = "llama3_gold_quant_checkpoint" 
DEFAULT_BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"
DEFAULT_NEWS_FILE = "final/gold_news_10years.csv"
DEFAULT_CACHE_FILE = "commodity_data/gold.csv"
DEFAULT_START_DATE = "2025-09-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_DOWNLOAD_END_DATE = "2026-01-10"

DEFAULT_OUTPUT_CSV = "q4_memory_legacy_results.csv"
DEFAULT_OUTPUT_CHART = "q4_memory_legacy_chart.png"
DEFAULT_ORIGINAL_STRATEGY_CSV = "final/q4_strategy_daily.csv"

# ================= 1. Helper Functions (Copied) =================
def compute_technical_indicators(df):
    # We keep this just to get a clean dataframe, though we won't feed indicators to LLM
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def get_next_trading_day(date, valid_dates):
    d = pd.Timestamp(date).normalize()
    idx = valid_dates.searchsorted(d)
    if idx >= len(valid_dates):
        return valid_dates[-1]
    return valid_dates[idx]

def prepare_daily_data(news_file, cache_file, start_date, end_date):
    print("Loading and preparing data...")
    df_news = pd.read_csv(news_file)
    df_news["Date"] = pd.to_datetime(df_news["Date"])
    
    # Load Market Data
    if os.path.exists(cache_file):
        gold = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        # Fallback simple download
        gold = yf.download("GC=F", start="2020-01-01", end=end_date, progress=False)
        if isinstance(gold.columns, pd.MultiIndex): gold.columns = gold.columns.get_level_values(0)
        gold.to_csv(cache_file)

    gold = compute_technical_indicators(gold)
    if gold.index.tz is not None: gold.index = gold.index.tz_localize(None)
    valid_dates = pd.DatetimeIndex(gold.index).normalize()

    # Align News (Standard Logic)
    df_news["Trading_Date"] = df_news["Date"].apply(lambda x: get_next_trading_day(x, valid_dates))

    daily_records = []
    daily_groups = df_news.groupby("Trading_Date")
    
    for date, group in daily_groups:
        if date not in gold.index: continue
        headlines = group["Headline"].tolist()
        row = gold.loc[date]
        
        # Note: No Technical String generation here
        
        daily_records.append({
            "Date": date, 
            "Headlines": headlines,
            "Full_Row": row
        })

    return pd.DataFrame(daily_records).set_index("Date").sort_index(), gold

# ================= 2. Memory & Backtest Logic =================
def run_memory_backtest(args):
    # 1. Data
    df_daily, gold_price = prepare_daily_data(args.news_file, args.cache_file, args.start_date, args.end_date)
    test_dates = df_daily[args.start_date:args.end_date].index

    # 2. Load Model
    print(f"Loading Base Model: {args.base_model}")
    print(f"Loading Adapter: {args.model_path}")
    
    # Load Base First
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model, # Load Llama-3 Base
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    
    # Load Adapter
    try:
        model.load_adapter(args.model_path)
        print("✅ Adapter loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load adapter from {args.model_path}: {e}")
        # Try finding absolute path if relative failed
        abs_path = os.path.abspath(args.model_path)
        print(f"   Attempting absolute path: {abs_path}")
        try:
            model.load_adapter(abs_path)
            print("✅ Adapter loaded successfully (absolute path)!")
        except:
            raise RuntimeError("Could not load the fine-tuned adapter. Check path.")

    FastLanguageModel.for_inference(model)

    results = []
    
    # MEMORY BUFFER: List of {"headline": str, "date": Timestamp}
    memory_buffer = [] 
    
    print(f"🚀 Running Legacy Memory Backtest ({len(test_dates)} days)...")

    for current_date in tqdm(test_dates):
        try:
            row = df_daily.loc[current_date]
        except KeyError: continue

        # --- A. PREPARE INPUT (With Memory Injection) ---
        
        # 1. Prune Memory (Remove > 14 days)
        valid_memory = []
        cutoff_date = current_date - timedelta(days=14)
        for mem in memory_buffer:
            if mem['date'] >= cutoff_date:
                valid_memory.append(mem)
        memory_buffer = valid_memory # Update buffer

        # 2. Inject Memory into News
        current_headlines = row['Headlines']
        
        # Create "Synthetic" headlines from memory
        memory_headlines = [f"[HISTORY {m['date'].strftime('%Y-%m-%d')}] {m['headline']}" for m in valid_memory]
        
        # Combine
        combined_headlines = current_headlines + memory_headlines
        
        # Construct Prompt - NEWS ONLY (Matching Legacy Format)
        # Format used in 3.ipynb/4.ipynb for training was:
        # "Date: YYYY-MM-DD\nNews: Headline" (for single headline)
        # BUT here we have MULTIPLE headlines per day.
        # How did 5.1.4 handle multiple? It iterated them one by one.
        # IF we want to feed ALL headlines at once to a model trained on single headlines,
        # it might fail or perform poorly.
        # However, Llama 3 is smart. Let's try feeding the list.
        # OR should we stick to "Single Day Aggregate"?
        # 5.1.4 aggregated SCORES, not inputs.
        
        # Wait, if the model was trained on SINGLE headlines, and we feed it a LIST, it is OOD.
        # BUT, the prompt format in 3.ipynb was:
        # "Date: ...\nNews: ..."
        
        # Let's construct a prompt that looks like a single "News" block containing the list.
        news_text = "\n".join([f"- {h}" for h in combined_headlines])
        input_text = f"Date: {current_date.strftime('%Y-%m-%d')}\nNews:\n{news_text}"

        # --- B. INFERENCE (Scoring) ---
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Macro Quant Strategist specializing in Gold (XAU/USD). Analyze the given news headline and assign a Sentiment Score from -5 to +5.

### Input:
{input_text}

### Response:
"""
        # Generate Score
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, temperature=0.05)
        out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Parse Score
        response_part = out_text.split("### Response:")[-1].strip()
        score = 0.0
        match = re.search(r"(?:Score:\s*)?([+\-]?\d+(\.\d+)?)", response_part)
        if match:
            score = float(match.group(1))
        
        results.append({"Date": current_date, "Score": score, "Memory_Size": len(valid_memory)})

        # --- C. UPDATE MEMORY (Summarization) ---
        # Summarize TODAY'S headlines only (not history)
        summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial news filter. Identify the single most important event from today's news that will impact Gold prices for the next 2 weeks.
If nothing is major, reply "None".
Format: "EVENT SUMMARY" (Max 15 words).<|eot_id|><|start_header_id|>user<|end_header_id|>

News:
{chr(10).join(current_headlines)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        with torch.no_grad():
            with model.disable_adapter():
                inputs_sum = tokenizer([summary_prompt], return_tensors="pt").to("cuda")
                outputs_sum = model.generate(**inputs_sum, max_new_tokens=48, temperature=0.1, use_cache=True)
                sum_text = tokenizer.batch_decode(outputs_sum, skip_special_tokens=True)[0]
                
                raw_summary = sum_text.split("assistant")[-1].strip()
                if "None" not in raw_summary and len(raw_summary) > 5:
                    clean_summary = raw_summary.replace('"', '').split('\n')[0]
                    memory_buffer.append({"headline": clean_summary, "date": current_date})

    # ================= 3. Analysis & Plotting =================
    df_res = pd.DataFrame(results).set_index("Date")
    
    # Returns
    gold_returns = gold_price["Close"].pct_change().shift(-1)
    strategy_df = df_res.join(gold_returns.rename("Gold_Ret"))
    
    # Signal
    strategy_df["Position"] = np.where(strategy_df["Score"] > 0, 1, -1)
    strategy_df["Position"] = np.where(strategy_df["Score"] == 0, 0, strategy_df["Position"])
    
    strategy_df["Strategy_Ret"] = strategy_df["Position"] * strategy_df["Gold_Ret"]
    strategy_df["Cum_Strategy"] = (1 + strategy_df["Strategy_Ret"].fillna(0)).cumprod()
    strategy_df["Cum_Gold"] = (1 + strategy_df["Gold_Ret"].fillna(0)).cumprod()
    
    # Metrics
    final_ret = strategy_df["Cum_Strategy"].iloc[-1] - 1
    
    # Max Drawdown
    cum_max = strategy_df["Cum_Strategy"].cummax()
    drawdown = (strategy_df["Cum_Strategy"] - cum_max) / cum_max
    max_drawdown = drawdown.min()
    
    print("\n" + "="*40)
    print(f"🚀 Legacy Memory Strategy Final Return: {final_ret*100:.2f}%")
    print(f"📉 Max Drawdown: {max_drawdown:.2%}")
    print("="*40)
    
    strategy_df.to_csv(args.output_csv)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df.index, strategy_df["Cum_Strategy"], label="Legacy Memory AI (News Only)", color="darkorange", linewidth=2)
    
    # Comparison
    if args.original_csv and os.path.exists(args.original_csv):
        try:
            print(f"Loading original strategy data from: {args.original_csv}")
            orig_df = pd.read_csv(args.original_csv)
            if "Date" in orig_df.columns:
                orig_df["Date"] = pd.to_datetime(orig_df["Date"])
                orig_df.set_index("Date", inplace=True)
            
            common_idx = strategy_df.index.intersection(orig_df.index)
            if not common_idx.empty:
                orig_aligned = orig_df.loc[common_idx]
                if "Cumulative_Strategy" in orig_aligned.columns:
                    o_cum = orig_aligned["Cumulative_Strategy"]
                    o_final = o_cum.iloc[-1] - 1
                    
                    print("-" * 20)
                    print(f"📉 Original Strategy (5.1.4)")
                    print(f"Final Return: {o_final*100:.2f}%")
                    print("-" * 20)
                    
                    plt.plot(orig_aligned.index, o_cum, label="Original 5.1.4", color="green", linestyle=":", linewidth=1.5)
        except Exception as e:
            print(f"Warning: Failed to load original CSV: {e}")

    plt.plot(strategy_df.index, strategy_df["Cum_Gold"], label="Gold Benchmark", color="gray", linestyle="--")
    plt.title("Gold Strategy (Legacy Model + Memory, No Technicals)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output_chart)
    print(f"✅ Chart saved to {args.output_chart}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--news-file", type=str, default=DEFAULT_NEWS_FILE)
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-chart", type=str, default=DEFAULT_OUTPUT_CHART)
    parser.add_argument("--original-csv", type=str, default=DEFAULT_ORIGINAL_STRATEGY_CSV)
    args = parser.parse_args()
    
    run_memory_backtest(args)
