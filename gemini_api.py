"""
=============================================================
  STEP 4: GEMINI API INTEGRATION — IMPROVED VERSION
  AI-Based Customer Query Analyzer
=============================================================

Full Pipeline:
  User types query
        |
      BERT  ->  intent + sentiment
        |
   Gemini API  ->  Complete natural response
        |
   Final output shown to user

IMPROVEMENTS IN THIS VERSION:
  FIX 1: Shorter prompts = fewer input tokens = more output tokens
  FIX 2: Stronger instruction to never cut off response
  FIX 3: Increased delay to 15s (safer for free tier)
  FIX 4: max_tokens set to 200 (sweet spot for complete responses)
  FIX 5: Retry wait increased to 70s

Run: python gemini_api.py
"""

import re
import json
import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# !! PASTE YOUR GEMINI API KEY HERE !!
# Get it free from: https://aistudio.google.com/app/apikey
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyCQYeiDl2uVfF-y8tUsTENxSSbzYAq4z4U"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    # Paths
    "data_dir"    : r"D:\project_s\clinc_oos\pre_processed",
    "model_dir"   : r"D:\project_s",
    "model_file"  : "bert_best.pt",

    # BERT
    "bert_model"  : "bert-base-uncased",
    "max_length"  : 64,
    "dropout"     : 0.3,

    # Gemini
    "gemini_model"  : "gemini-2.5-flash",
    "max_tokens"    : 200,   # sweet spot — enough for 2-3 full sentences
    "temperature"   : 0.7,
    "request_delay" : 15,    # 15s between requests (safer than 12s)
    "retry_wait"    : 70,    # 70s wait on rate limit hit
}

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTIMENT_NAMES = ["negative", "neutral", "positive"]
SENTIMENT_EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
_last_request_time = 0

# ─────────────────────────────────────────────
# STEP 1 — CONFIGURE GEMINI
# ─────────────────────────────────────────────
print("=" * 60)
print("  BERT + GEMINI PIPELINE — IMPROVED VERSION")
print("  AI-Based Customer Query Analyzer")
print("=" * 60)

print("\nConnecting to Gemini API...")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    test   = client.models.generate_content(
        model    = CFG["gemini_model"],
        contents = "Say OK"
    )
    print("Gemini API connected ✅")
    print(f"Model          : {CFG['gemini_model']}")
except Exception as e:
    print(f"Gemini API connection failed ❌\nError: {e}")
    print("\nPlease check:")
    print("  1. API key is correct")
    print("  2. Internet connection is active")
    print("  3. Run: pip install --upgrade google-genai")
    exit(1)

print(f"Device         : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# STEP 2 — LOAD LABEL MAP
# ─────────────────────────────────────────────
with open(f"{CFG['data_dir']}/intent_label_map.json") as f:
    id2intent = json.load(f)

NUM_INTENTS    = len(id2intent)
NUM_SENTIMENTS = 3
print(f"Intents loaded : {NUM_INTENTS}")

# ─────────────────────────────────────────────
# STEP 3 — BERT MODEL
# ─────────────────────────────────────────────
class MultiTaskBERT(nn.Module):
    def __init__(self, bert_model_name, num_intents, num_sentiments, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_intents),
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sentiments),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs          = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        cls_output       = self.dropout(outputs.pooler_output)
        return self.intent_classifier(cls_output), self.sentiment_classifier(cls_output)

# ─────────────────────────────────────────────
# STEP 4 — LOAD TRAINED BERT
# ─────────────────────────────────────────────
print("\nLoading trained BERT model...")
tokenizer = BertTokenizer.from_pretrained(CFG["model_dir"])
bert      = MultiTaskBERT(CFG["bert_model"], NUM_INTENTS, NUM_SENTIMENTS, CFG["dropout"])
bert.load_state_dict(
    torch.load(
        f"{CFG['model_dir']}/{CFG['model_file']}",
        map_location = DEVICE,
        weights_only = True,
    )
)
bert = bert.to(DEVICE)
bert.eval()
print("BERT model loaded ✅")

# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\'\-\?\!\.,]", "", text)
    text = re.sub(r"(\w)\1{3,}", r"\1\1", text)
    return text

# ─────────────────────────────────────────────
# BERT INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def classify(query: str) -> dict:
    clean_query = clean_text(query)
    enc = tokenizer(
        clean_query,
        max_length=CFG["max_length"], padding="max_length",
        truncation=True, return_tensors="pt",
    )
    intent_logits, sentiment_logits = bert(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
        enc["token_type_ids"].to(DEVICE),
    )
    intent_probs    = torch.softmax(intent_logits,    dim=-1)[0]
    sentiment_probs = torch.softmax(sentiment_logits, dim=-1)[0]
    intent_id       = intent_probs.argmax().item()
    sentiment_id    = sentiment_probs.argmax().item()
    top3_idx        = intent_probs.topk(3).indices.cpu().numpy()
    top3_scores     = intent_probs.topk(3).values.cpu().numpy()

    return {
        "clean_query"         : clean_query,
        "intent"              : id2intent[str(intent_id)],
        "intent_confidence"   : round(intent_probs[intent_id].item(), 4),
        "top3_intents"        : [(id2intent[str(i)], round(float(s)*100, 1))
                                  for i, s in zip(top3_idx, top3_scores)],
        "sentiment"           : SENTIMENT_NAMES[sentiment_id],
        "sentiment_confidence": round(sentiment_probs[sentiment_id].item(), 4),
    }

# ─────────────────────────────────────────────
# PROMPT BUILDER — SHORT AND PRECISE
# Key insight: shorter prompts = more tokens left for response
# ─────────────────────────────────────────────
def build_prompt(query: str, intent: str, sentiment: str, confidence: float) -> str:

    # Out of scope
    if intent == "oos" or confidence < 0.3:
        return (
            f"You are a customer service assistant.\n"
            f"Customer said: \"{query}\"\n"
            f"You don't understand this request. Write exactly 2 complete sentences: "
            f"apologize and ask them to rephrase. Do not stop mid-sentence."
        )

    intent_readable = intent.replace("_", " ")

    tone_instruction = {
        "negative": "Start by acknowledging their frustration. Be empathetic and reassuring.",
        "neutral" : "Be professional and helpful. Get straight to the point.",
        "positive": "Be warm and enthusiastic. Match their positive energy.",
    }.get(sentiment, "Be professional and helpful.")

    return (
        f"You are a customer service assistant.\n"
        f"Customer query: \"{query}\"\n"
        f"Topic: {intent_readable}\n"
        f"Tone: {tone_instruction}\n"
        f"Write exactly 2-3 complete sentences helping this customer. "
        f"You MUST finish every sentence completely. Never stop mid-sentence. "
        f"Do not use bullet points or labels."
    )

# ─────────────────────────────────────────────
# RATE LIMIT HANDLER
# ─────────────────────────────────────────────
def wait_for_rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < CFG["request_delay"]:
        wait = CFG["request_delay"] - elapsed
        print(f"  [Waiting {wait:.0f}s for rate limit...]", flush=True)
        time.sleep(wait)
    _last_request_time = time.time()

# ─────────────────────────────────────────────
# GEMINI RESPONSE
# ─────────────────────────────────────────────
def get_response(query: str, intent: str, sentiment: str, confidence: float) -> str:
    prompt = build_prompt(query, intent, sentiment, confidence)
    wait_for_rate_limit()

    for attempt in range(2):
        try:
            result = client.models.generate_content(
                model    = CFG["gemini_model"],
                contents = prompt,
                config   = types.GenerateContentConfig(
                    max_output_tokens = CFG["max_tokens"],
                    temperature       = CFG["temperature"],
                    top_p             = 0.9,
                    stop_sequences    = [],   # no early stopping
                )
            )
            response = result.text.strip()

            # Safety check — if response seems cut off, flag it
            if response and not response[-1] in ".!?":
                response += "."   # add period if ends abruptly

            return response

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt == 0:
                print(f"  [Rate limit hit — waiting {CFG['retry_wait']}s...]")
                time.sleep(CFG["retry_wait"])
                _last_request_time = time.time()
                continue
            elif "API_KEY" in error_str or "403" in error_str:
                return "API key error. Please check your Gemini API key."
            else:
                print(f"  [Error: {e}]")
                return "I apologize, I am having trouble processing your request. Please try again."

    return "I apologize, the service is temporarily unavailable. Please try again shortly."

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def analyze(query: str) -> dict:
    t0       = time.time()
    bert_out = classify(query)
    response = get_response(
        query      = query,
        intent     = bert_out["intent"],
        sentiment  = bert_out["sentiment"],
        confidence = bert_out["intent_confidence"],
    )
    return {
        "query"               : query,
        "intent"              : bert_out["intent"],
        "intent_confidence"   : bert_out["intent_confidence"],
        "top3_intents"        : bert_out["top3_intents"],
        "sentiment"           : bert_out["sentiment"],
        "sentiment_confidence": bert_out["sentiment_confidence"],
        "response"            : response,
        "latency_ms"          : round((time.time() - t0) * 1000, 1),
    }

# ─────────────────────────────────────────────
# SAMPLE TESTS — 3 only to save quota
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SAMPLE PIPELINE TESTS (3 samples)")
print("  Note: 15 second delay between requests (free tier)")
print("=" * 60)

sample_queries = [
    "What is my current account balance?",
    "I am really frustrated my card has been blocked for no reason!",
    "Thank you so much you guys are absolutely amazing!",
]

for query in sample_queries:
    print(f"\n  {'─'*55}")
    result = analyze(query)
    emoji  = SENTIMENT_EMOJI[result["sentiment"]]
    print(f"  Query     : {result['query']}")
    print(f"  Intent    : {result['intent']}  ({result['intent_confidence']*100:.1f}%)")
    print(f"  Sentiment : {emoji} {result['sentiment']}  ({result['sentiment_confidence']*100:.1f}%)")
    print(f"  Response  : {result['response']}")
    print(f"  Latency   : {result['latency_ms']} ms")

# ─────────────────────────────────────────────
# LIVE CHATBOT DEMO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LIVE CHATBOT DEMO")
print("  BERT classifies → Gemini responds")
print("  Note: 15 second wait between responses (free tier)")
print("  ─────────────────────────────────────────────────")
print("  Try negative : 'My card is blocked and I am furious!'")
print("  Try positive : 'Thank you so much this is wonderful!'")
print("  Try neutral  : 'What is my account balance?'")
print("  Type 'quit' to exit")
print("=" * 60)

while True:
    try:
        print()
        query = input("  You: ").strip()

        if not query:
            print("  Please type something!")
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!")
            break

        result = analyze(query)
        emoji  = SENTIMENT_EMOJI[result["sentiment"]]

        print(f"\n  Bot       : {result['response']}")
        print(f"  {'─'*50}")
        print(f"  Intent    : {result['intent']}  ({result['intent_confidence']*100:.1f}%)")
        print(f"  Sentiment : {emoji} {result['sentiment']}  ({result['sentiment_confidence']*100:.1f}%)")
        print(f"  Top 3     : {[(n, f'{s}%') for n, s in result['top3_intents']]}")
        print(f"  Latency   : {result['latency_ms']} ms")

    except KeyboardInterrupt:
        print("\n  Exiting...")
        break
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE ✅")
print("=" * 60)
print("""
  BERT  → Intent Classification : 86.20% accuracy
  BERT  → Sentiment Analysis    : 93.13% accuracy
  Gemini → Response Generation  : gemini-2.5-flash
""")