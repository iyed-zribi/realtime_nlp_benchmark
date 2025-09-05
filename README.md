#  NLP Model Benchmark on Reddit or Custom Text

This project benchmarks different NLP models on:
- Reddit live data
- or your own custom text

It measures:
- Model load time
- Average inference time per text
- Memory usage

It also shows **interactive graphs** (Plotly) and exports results to a downloadable **CSV file**.

---

## Features
- Select pretrained Hugging Face models from a dropdown
- Add your **own trained model** via path
- Input **custom text** (instead of Reddit)
- Benchmark and compare performance
- Download results as **CSV**
- Interactive graphs for inference time & memory usage
- Highlights the **best overall model** (fastest + most memory-efficient)

---

## Installation

Clone repo:
```bash
git clone https://github.com/iyed-zribi/realtime_nlp_benchmark.git
cd realtime_nlp_benchmark
