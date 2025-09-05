# Install dependencies if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    !pip install gradio asyncpraw transformers torch matplotlib nest_asyncio pandas psutil plotly


import gradio as gr
import asyncio
import nest_asyncio
nest_asyncio.apply()

import asyncpraw
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import psutil, os, time
import plotly.express as px

# --------- BENCHMARK FUNCTION ---------
def benchmark_model(model_name, texts):
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    load_time = time.time() - start_load

    start_infer = time.time()
    results = [nlp(t[:200])[0] for t in texts]  # truncate to avoid token limits
    infer_time = (time.time() - start_infer) / len(texts)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
    return {
        "Model": model_name,
        "Load Time (s)": round(load_time,2),
        "Avg Inference Time (s/text)": round(infer_time,4),
        "Memory Usage (MB)": round(mem_usage,2)
    }

# --------- FETCH REDDIT DATA ---------
async def fetch_sample(CLIENT_ID, CLIENT_SECRET, USER_AGENT, subreddit_name="technology+news", limit=5):
    reddit = asyncpraw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    subreddit = await reddit.subreddit(subreddit_name)
    texts = []
    async for submission in subreddit.new(limit=limit):
        texts.append(submission.title + " " + submission.selftext)
    async for comment in subreddit.comments(limit=limit):
        texts.append(comment.body)
    return texts

# --------- GRADIO FUNCTION ---------
def run_benchmark(client_id, client_secret, user_agent, models, custom_model, custom_text, subreddit, limit):
    if not models and not custom_model:
        return "Please select at least one model or provide a custom model path.", None, None, None

    # If custom text provided, use it; else fetch Reddit data
    if custom_text.strip():
        texts = [custom_text.strip()]
    else:
        texts = asyncio.get_event_loop().run_until_complete(
            fetch_sample(client_id, client_secret, user_agent, subreddit, limit)
        )

    # Combine models + optional custom model
    all_models = list(models) if models else []
    if custom_model.strip():
        all_models.append(custom_model.strip())

    results = [benchmark_model(m, texts) for m in all_models]
    df = pd.DataFrame(results)

    # Overall best model (lowest normalized score = best balance of speed & efficiency)
    df_norm = df.copy()
    df_norm["Time_norm"] = df_norm["Avg Inference Time (s/text)"] / df_norm["Avg Inference Time (s/text)"].max()
    df_norm["Mem_norm"] = df_norm["Memory Usage (MB)"] / df_norm["Memory Usage (MB)"].max()
    df_norm["Score"] = df_norm["Time_norm"] + df_norm["Mem_norm"]
    best_overall = df_norm.loc[df_norm["Score"].idxmin()]

    summary = f"""
     Best Overall Model: **{best_overall['Model']}**  
     Avg Inference Time: {best_overall['Avg Inference Time (s/text)']}s/text  
     Memory Usage: {best_overall['Memory Usage (MB)']} MB  
    """

    # Save downloadable CSV (semicolon-separated)
    csv_path = "/tmp/benchmark_results.csv"
    df.to_csv(csv_path, sep=";", index=False)

    # Interactive graphs with Plotly
    fig_time = px.bar(df, x="Model", y="Avg Inference Time (s/text)", title="Average Inference Time (s/text)", text="Avg Inference Time (s/text)")
    fig_time.update_traces(textposition="outside")

    fig_mem = px.bar(df, x="Model", y="Memory Usage (MB)", title="Memory Usage (MB)", text="Memory Usage (MB)")
    fig_mem.update_traces(textposition="outside")

    return df.to_html(), summary, csv_path, fig_time, fig_mem

# --------- AVAILABLE MODELS ---------
models_available = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base-uncased",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # 🔽 you can add more pretrained models here
]

# --------- GRADIO INTERFACE ---------
iface = gr.Interface(
    fn=run_benchmark,
    inputs=[
        gr.Textbox(label="Reddit Client ID"),
        gr.Textbox(label="Reddit Client Secret"),
        gr.Textbox(value="hackathon-bot", label="User Agent"),
        gr.Dropdown(models_available, multiselect=True, label="Select Pretrained Models"),
        gr.Textbox(label="Custom Model Path (optional)"),
        gr.Textbox(label="Custom Text (optional, overrides Reddit)"),
        gr.Textbox("technology+news", label="Subreddit"),
        gr.Slider(1,20,5, label="Number of posts/comments")
    ],
    outputs=[
        gr.HTML(label="Benchmark Results"),
        gr.Markdown(label="Best Model Summary"),
        gr.File(label="Download CSV"),
        gr.Plot(label="Inference Time"),
        gr.Plot(label="Memory Usage")
    ],
    title="NLP Model Benchmark"
)

iface.launch()
