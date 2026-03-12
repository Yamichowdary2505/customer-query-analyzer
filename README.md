# AI-Based Customer Query Analyzer

An intelligent customer service chatbot powered by a fine-tuned BERT multi-task model for intent classification and sentiment analysis, with LLM-generated responses via Groq / Gemini / OpenAI / Claude.

---

## Project Overview

| Component | Details |
|---|---|
| Intent Classification | Fine-tuned BERT (bert-base-uncased) — 151 intents |
| Sentiment Analysis | Multi-task BERT head — negative / neutral / positive |
| Dataset | CLINC150 (clinc/clinc_oos, "plus" variant) |
| Response Generation | Groq (LLaMA 3.1) / Gemini / OpenAI / Claude |
| Frontend | Streamlit dashboard with real-time analytics |

---

## Model Performance

| Task | Accuracy | F1 Score |
|---|---|---|
| Intent Classification | **86.20%** | 0.8502 |
| Sentiment Analysis | **93.13%** | 0.9263 |

- Training: 15 epochs on GPU, best checkpoint at epoch 10
- Architecture: BERT backbone + dual classification heads

---

## Pipeline

```
Customer Query
    |
    v
Safety Net (pre-classification)
    |  Catches: fraud, unauthorized access, emergency block, compromised account
    |  Uses keyword matching for intents NOT in CLINC150 dataset
    v
BERT Multi-Task Model
    |  Intent Classification  -> 151 classes
    |  Sentiment Analysis     -> negative / neutral / positive
    v
Prompt Builder
    |  Sentiment-aware tone + Intent-specific guidance
    |  Conversation history context (last 4 turns)
    v
LLM Response (Groq / Gemini / OpenAI / Claude)
    |
    v
Streamlit Dashboard
```

---

## Safety Net

CLINC150 does not contain security-critical intents like fraud or unauthorized access. A pre-classification safety layer was implemented as a deliberate design decision for high-priority queries outside the training distribution.

This is consistent with industry-standard hybrid NLU architectures (e.g. BofA Erica, PayPal, Alexa).

Covered intents:
- `unauthorized_access` — someone using account without permission
- `report_fraud` — unauthorized transactions / scams
- `emergency_block` — lost or stolen card
- `account_compromised` — locked out / password changed

---

## Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-query-analyzer.git
cd customer-query-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your model files
Place these files in the paths configured in the sidebar:
- `bert_best.pt` — trained BERT model weights
- `intent_label_map.json` — intent ID to label mapping
- BERT tokenizer files: `vocab.txt`, `tokenizer.json`, `tokenizer_config.json`

Note: These files are NOT included in this repo (too large for GitHub).
Train the model using `train.py` or obtain from the project owner.

### 4. Get a free API key
Recommended: Groq (free, fast) — https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

Note: Streamlit Cloud runs on Linux servers without access to your local model files.
The app requires the BERT model (bert_best.pt) and tokenizer files to function.
For cloud deployment, host these files externally (e.g. Google Drive, HuggingFace Hub)
and modify the loading code accordingly.

Steps:
1. Push this repository to GitHub
2. Go to https://streamlit.io/cloud — New app
3. Connect your GitHub repository
4. Set Main file path to `app.py`
5. Click Deploy

---

## Project Structure

```
customer-query-analyzer/
|
|-- app.py              <- Streamlit dashboard (main app)
|-- API.py              <- Terminal chatbot (local testing)
|-- pre_process.py      <- Data preprocessing
|-- train.py            <- BERT training script
|-- test.py             <- Model evaluation
|-- requirements.txt    <- Python dependencies
|-- README.md           <- This file
|-- .gitignore          <- Excludes model weights, API keys, datasets
```

---

## Configuration

In the app.py sidebar you can configure:
- AI Provider: groq / gemini / openai / claude
- API Key: paste your key (never stored, session-only)
- BERT Model Path: folder containing bert_best.pt and tokenizer files
- Data Path: folder containing intent_label_map.json

---

## Dashboard Features

- Chat Interface — real-time chat with BERT + LLM pipeline
- Analytics Panel — intent confidence bars, sentiment breakdown chart
- Session Overview — sentiment distribution donut chart
- Query History — full log of all queries with intent, confidence, latency
- Security Alerts — visual indicator for fraud/security queries
- Low Confidence — flags queries BERT is uncertain about

---

## Technologies Used

- PyTorch — model training and inference
- HuggingFace Transformers — BERT tokenizer and base model
- CLINC150 Dataset — intent classification dataset
- Streamlit — web dashboard
- Plotly — interactive charts
- Groq API — LLM response generation

---

## Author

Built as part of an AI-based NLP project.
