"""
=============================================================
  STEP 4: MULTI-PROVIDER AI INTEGRATION
  AI-Based Customer Query Analyzer
=============================================================

Supports 4 AI providers — just set your API key below:

  Provider 1 : Groq API     — FREE, instant, no quota issues ✅ RECOMMENDED
  Provider 2 : Google Gemini — free tier (slow, quota limits)
  Provider 3 : OpenAI ChatGPT — paid (~$0.002/query)
  Provider 4 : Anthropic Claude — paid (~$0.003/query)

HOW TO USE:
  1. Set ACTIVE_PROVIDER to "groq", "gemini", "openai", or "claude"
  2. Paste the corresponding API key
  3. Run: python API.py

GET API KEYS (all free to sign up):
  Groq   : https://console.groq.com              (FREE — recommended)
  Gemini : https://aistudio.google.com/app/apikey (free tier)
  OpenAI : https://platform.openai.com/api-keys   ($5 free credits)
  Claude : https://console.anthropic.com           ($5 free credits)

Run: python API.py
"""

import re
import json
import time
import requests
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# =============================================================
#   PROVIDER SELECTION — CHANGE THIS TO SWITCH PROVIDER
# =============================================================
ACTIVE_PROVIDER = "groq"      # "groq" / "gemini" / "openai" / "claude"

# Paste your API key for the provider you want to use
API_KEYS = {
    "groq"  : "YOUR_GROQ_API_KEY_HERE",    # from console.groq.com (FREE)
    "gemini": "YOUR_GEMINI_API_KEY_HERE",  # from aistudio.google.com
    "openai": "YOUR_OPENAI_API_KEY_HERE",  # from platform.openai.com
    "claude": "YOUR_CLAUDE_API_KEY_HERE",  # from console.anthropic.com
}

# Models used per provider
MODELS = {
    "groq"  : "llama-3.1-8b-instant",      # fast, free, excellent quality
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
}

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    # Paths
    "data_dir"  : r"D:\project_s\clinc_oos\pre_processed",
    "model_dir" : r"D:\project_s",
    "model_file": "bert_best.pt",

    # BERT
    "bert_model": "bert-base-uncased",
    "max_length": 64,
    "dropout"   : 0.3,

    # Generation
    "max_tokens"         : 200,
    "temperature"        : 0.7,
    "low_conf_threshold" : 0.20,   # below this → OOS fallback (0.20 suits 151 intents)

    # Delay between requests per provider (seconds)
    "delays": {
        "groq"  : 1,    # free and fast — almost no delay needed
        "gemini": 20,   # free tier — needs delay
        "openai": 1,    # paid — instant
        "claude": 1,    # paid — instant
    }
}

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTIMENT_NAMES = ["negative", "neutral", "positive"]
SENTIMENT_EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
_last_request_time = 0

# ─────────────────────────────────────────────
# STEP 1 — VALIDATE PROVIDER
# ─────────────────────────────────────────────
print("=" * 60)
print("  BERT + AI PIPELINE — MULTI PROVIDER VERSION")
print("  AI-Based Customer Query Analyzer")
print("=" * 60)

if ACTIVE_PROVIDER not in API_KEYS:
    print(f"❌ Unknown provider: {ACTIVE_PROVIDER}")
    print("   Set ACTIVE_PROVIDER to: groq / gemini / openai / claude")
    exit(1)

ACTIVE_KEY    = API_KEYS[ACTIVE_PROVIDER]
ACTIVE_MODEL  = MODELS[ACTIVE_PROVIDER]
REQUEST_DELAY = CFG["delays"][ACTIVE_PROVIDER]

if "YOUR_" in ACTIVE_KEY:
    print(f"❌ Please paste your {ACTIVE_PROVIDER.upper()} API key in the script!")
    print(f"   API_KEYS['{ACTIVE_PROVIDER}'] = 'YOUR_KEY_HERE'  ← replace this")
    if ACTIVE_PROVIDER == "groq":
        print("   Get your FREE key at: https://console.groq.com")
    exit(1)

print(f"\nProvider       : {ACTIVE_PROVIDER.upper()}")
print(f"Model          : {ACTIVE_MODEL}")

# ─────────────────────────────────────────────
# STEP 2 — TEST API CONNECTION
# ─────────────────────────────────────────────
print("Connecting to API...")

def test_connection() -> bool:
    try:
        if ACTIVE_PROVIDER == "groq":
            url     = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {ACTIVE_KEY}", "Content-Type": "application/json"}
            body    = {"model": ACTIVE_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
            r = requests.post(url, headers=headers, json=body, timeout=15)
            return r.status_code == 200

        elif ACTIVE_PROVIDER == "gemini":
            url  = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent?key={ACTIVE_KEY}"
            body = {"contents": [{"parts": [{"text": "Say OK"}]}], "generationConfig": {"maxOutputTokens": 5}}
            r = requests.post(url, json=body, timeout=15)
            return r.status_code == 200

        elif ACTIVE_PROVIDER == "openai":
            url     = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {ACTIVE_KEY}", "Content-Type": "application/json"}
            body    = {"model": ACTIVE_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
            r = requests.post(url, headers=headers, json=body, timeout=15)
            return r.status_code == 200

        elif ACTIVE_PROVIDER == "claude":
            url     = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": ACTIVE_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
            body    = {"model": ACTIVE_MODEL, "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK"}]}
            r = requests.post(url, headers=headers, json=body, timeout=15)
            return r.status_code == 200

    except Exception as e:
        print(f"  Connection error: {e}")
        return False

if test_connection():
    print(f"API connected ✅")
else:
    print(f"API connection failed ❌")
    print("Please check your API key and internet connection.")
    exit(1)

print(f"Device         : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# STEP 3 — LOAD LABEL MAP
# ─────────────────────────────────────────────
with open(f"{CFG['data_dir']}/intent_label_map.json") as f:
    id2intent = json.load(f)

NUM_INTENTS    = len(id2intent)
NUM_SENTIMENTS = 3
print(f"Intents loaded : {NUM_INTENTS}")

# from pipeline.py — find OOS label id for low confidence fallback
OOS_ID = next((int(k) for k, v in id2intent.items() if v == "oos"), -1)

# ─────────────────────────────────────────────
# STEP 4 — BERT MODEL DEFINITION
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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = self.dropout(outputs.pooler_output)
        return self.intent_classifier(cls), self.sentiment_classifier(cls)

# ─────────────────────────────────────────────
# STEP 5 — LOAD TRAINED BERT
# ─────────────────────────────────────────────
print("\nLoading trained BERT model...")
tokenizer = BertTokenizer.from_pretrained(CFG["model_dir"])
bert      = MultiTaskBERT(CFG["bert_model"], NUM_INTENTS, NUM_SENTIMENTS, CFG["dropout"])
bert.load_state_dict(
    torch.load(
        f"{CFG['model_dir']}/{CFG['model_file']}",
        map_location=DEVICE,
        weights_only=True,
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
# PRE-CLASSIFICATION SAFETY NET
# Catches high-stakes queries BERT misses due to
# dataset limitations (CLINC150 has no unauthorized
# access / emergency intent)
# Runs BEFORE BERT — overrides intent if matched
# ─────────────────────────────────────────────
SAFETY_PATTERNS = {
    "unauthorized_access": [
        "someone else", "someone is using", "unauthori", "hacked", "hack",
        "not me", "wasn't me", "i didn't do", "suspicious login", "unknown login",
        "someone logged", "someone accessed", "my account was accessed",
        "strange activity", "unusual activity", "unknown transaction",
        "i didn't make this", "i did not make", "fraudulent login",
    ],
    "report_fraud": [
        "fraud", "scam", "scammed", "cheated", "stolen", "stole", "theft",
        "fake transaction", "unauthorized transaction", "didn't authorize",
        "did not authorize", "money missing", "money gone", "money disappeared",
        "deducted without", "charged without", "debited without my",
    ],
    "emergency_block": [
        "block immediately", "block my card now", "freeze immediately",
        "lost my card", "card stolen", "stolen card", "i lost my",
        "cant find my card", "missing card", "card is missing",
    ],
    "account_compromised": [
        "account compromised", "account breached", "password changed",
        "someone changed my password", "locked out", "cant access my account",
        "cant log in", "cant login", "login not working", "otp not received",
        "not receiving otp", "verification not working",
    ],
}

def pre_classify(query: str):
    """
    Scans query for high-stakes keywords BEFORE BERT.
    Returns (intent_override, confidence) or (None, None) if no match.
    """
    q = query.lower()
    for intent, keywords in SAFETY_PATTERNS.items():
        for kw in keywords:
            if kw in q:
                return intent, 0.95   # high confidence since keyword matched
    return None, None

# ─────────────────────────────────────────────
# BERT INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def classify(query: str) -> dict:
    clean_query = clean_text(query)

    # ── Pre-classification safety net ──────────
    override_intent, override_conf = pre_classify(clean_query)

    enc = tokenizer(
        clean_query,
        max_length=CFG["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
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
    intent_conf     = intent_probs[intent_id].item()

    # ── Apply override if safety net matched ───
    if override_intent:
        intent_name = override_intent
        intent_conf = override_conf
        low_conf    = False
    # ── Low confidence → OOS fallback ──────────
    elif intent_conf < CFG["low_conf_threshold"] and OOS_ID >= 0:
        intent_name = "out_of_scope"
        low_conf    = True
    else:
        intent_name = id2intent[str(intent_id)]
        low_conf    = False

    top3_idx    = intent_probs.topk(3).indices.cpu().numpy()
    top3_scores = intent_probs.topk(3).values.cpu().numpy()

    return {
        "clean_query"         : clean_query,
        "intent"              : intent_name,
        "intent_confidence"   : round(intent_conf, 4),
        "top3_intents"        : [(id2intent[str(i)], round(float(s)*100, 1))
                                  for i, s in zip(top3_idx, top3_scores)],
        "sentiment"           : SENTIMENT_NAMES[sentiment_id],
        "sentiment_confidence": round(sentiment_probs[sentiment_id].item(), 4),
        "low_confidence"      : low_conf,
        "pre_classified"      : override_intent is not None,
    }

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
def build_prompt(query: str, intent: str, sentiment: str,
                 confidence: float, history: list = None) -> str:

    # ── OOS / Low confidence fallback ──────────
    if intent in ("oos", "out_of_scope") or confidence < CFG["low_conf_threshold"]:

        # If there is conversation history, the short reply is likely a follow-up
        # Pass the context so AI can answer it intelligently instead of asking to rephrase
        if history and len(history) > 0:
            context = "\nPrevious conversation:\n"
            for turn in history[-4:]:
                role     = "Customer" if turn["role"] == "user" else "Bot"
                context += f"  {role}: {turn['content']}\n"
            return (
                f"You are a professional customer service chatbot.\n"
                f"{context}\n"
                f"Customer's latest message: \"{query}\"\n\n"
                f"Instructions:\n"
                f"- The customer's message is short or unclear on its own\n"
                f"- But use the conversation history above to understand what they mean\n"
                f"- Respond naturally as if continuing the conversation\n"
                f"- Do NOT ask them to rephrase if the context makes it clear what they need\n"
                f"- Write 2-3 complete helpful sentences\n"
                f"- Never mention intent names, confidence scores or system labels"
            )

        # No history — genuinely unclear query, ask to rephrase
        return (
            f"You are a helpful customer service chatbot.\n"
            f"Customer said: \"{query}\"\n"
            f"You could not confidently understand this request.\n"
            f"Instructions:\n"
            f"- Apologize briefly and warmly\n"
            f"- Ask the customer to rephrase or provide more details\n"
            f"- Suggest they can ask about topics like account, payments, cards, orders, or bookings\n"
            f"- Write exactly 2-3 complete sentences\n"
            f"- Never mention confidence scores or intent labels"
        )

    intent_readable = intent.replace("_", " ")

    # ── Sentiment-based tone ────────────────────
    sentiment_tone = {
        "negative": (
            "The customer is frustrated, upset or angry. "
            "Open with a genuine apology and show you truly understand their frustration. "
            "Be calm, reassuring and solution-focused. "
            "Never be dismissive or make the customer feel at fault. "
            "Use soft, empathetic language throughout."
        ),
        "neutral": (
            "The customer is making a calm, factual request. "
            "Be professional, clear and concise. "
            "Get straight to the point without unnecessary filler. "
            "Provide accurate and actionable information."
        ),
        "positive": (
            "The customer is happy, excited or appreciative. "
            "Match their positive energy with warmth and enthusiasm. "
            "Be friendly and personable, not robotic. "
            "Make them feel valued and appreciated."
        ),
    }.get(sentiment, "Be professional, helpful and polite.")

    # ── Intent-based behaviour ──────────────────
    # Groups intents into categories and gives specific handling instructions
    intent_guidance = ""

    # ── SAFETY NET INTENTS — highest priority ──
    if intent == "unauthorized_access":
        intent_guidance = (
            "URGENT: Someone may be accessing the customer's account without permission. "
            "This is a critical security situation. "
            "Immediately advise them to: 1) Change their password right now, "
            "2) Enable two-factor authentication, "
            "3) Review recent login activity, "
            "4) Contact the security team if suspicious logins are found. "
            "Be urgent, clear and reassuring. Tell them their account security is the top priority."
        )
    elif intent == "report_fraud":
        intent_guidance = (
            "URGENT: The customer is reporting a fraud or unauthorized transaction. "
            "Treat this with the highest urgency. "
            "Immediately advise them to: 1) Block their card if involved, "
            "2) File a dispute for the unauthorized transaction, "
            "3) Note the transaction details for investigation. "
            "Reassure them they are protected and the transaction will be investigated. "
            "Be empathetic and action-oriented."
        )
    elif intent == "emergency_block":
        intent_guidance = (
            "URGENT: The customer needs to block their card immediately due to loss or theft. "
            "Advise them to block the card instantly via the app or by calling the helpline. "
            "Reassure them that once blocked, no transactions can be made. "
            "Offer to help them request a replacement card."
        )
    elif intent == "account_compromised":
        intent_guidance = (
            "URGENT: The customer cannot access their account or suspects it is compromised. "
            "Guide them to reset their password immediately via the official website or app. "
            "Advise them to check if their registered email or phone number has been changed. "
            "Suggest contacting support immediately if they are fully locked out."
        )
    # Account & Balance
    if any(x in intent for x in ["balance", "account", "bank", "statement"]):
        intent_guidance = (
            "Acknowledge the request. Explain you can help with account information. "
            "Guide them to check via app/website or offer to assist directly. "
            "Remind them to keep their login credentials secure."
        )
    # Card issues
    elif any(x in intent for x in ["card", "freeze", "block", "pin", "chip"]):
        intent_guidance = (
            "Take card issues seriously as they affect the customer's ability to access funds. "
            "Provide clear next steps such as blocking, replacing or unlocking the card. "
            "If urgent, prioritize immediate action steps."
        )
    # Payments & Transfers
    elif any(x in intent for x in ["transfer", "transaction", "payment", "send", "wire", "zelle", "venmo"]):
        intent_guidance = (
            "Be precise with payment-related queries as they involve money. "
            "Confirm the details of the transaction type. "
            "Provide processing times and any relevant limits or fees."
        )
    # Fraud & Security
    elif any(x in intent for x in ["fraud", "dispute", "unauthorized", "stolen", "scam", "chargeback"]):
        intent_guidance = (
            "Treat fraud and security issues with the highest urgency. "
            "Reassure the customer they are protected. "
            "Immediately guide them to report, block or dispute as needed. "
            "Express that their security is the top priority."
        )
    # Orders & Delivery
    elif any(x in intent for x in ["order", "delivery", "shipping", "track", "package", "cancel", "return", "refund"]):
        intent_guidance = (
            "Acknowledge the order or delivery concern clearly. "
            "Provide tracking steps or estimated resolution timeline. "
            "If there is a problem with the order, show willingness to resolve it promptly."
        )
    # Bookings & Reservations
    elif any(x in intent for x in ["book", "flight", "hotel", "reservation", "ticket", "travel", "uber", "lyft"]):
        intent_guidance = (
            "Help with the booking or reservation request clearly. "
            "Confirm details like date, destination or service type if needed. "
            "Provide next steps to complete or modify the booking."
        )
    # Loans & Credit
    elif any(x in intent for x in ["loan", "credit", "mortgage", "interest", "borrow", "emi", "debt"]):
        intent_guidance = (
            "Be informative and transparent about loan or credit topics. "
            "Provide eligibility, rates or process information where applicable. "
            "Avoid making promises — use phrases like 'typically' or 'generally'."
        )
    # Technical & App issues
    elif any(x in intent for x in ["app", "login", "password", "access", "reset", "otp", "verify", "2fa"]):
        intent_guidance = (
            "Guide the customer through technical steps clearly and simply. "
            "Use numbered steps if multiple actions are needed. "
            "Reassure them the issue is common and easily resolved."
        )
    # Complaints
    elif any(x in intent for x in ["complaint", "problem", "issue", "wrong", "error", "fail", "broken", "damaged"]):
        intent_guidance = (
            "Acknowledge the complaint with full empathy and without making excuses. "
            "Take ownership of the issue on behalf of the company. "
            "Offer a clear resolution path or escalation option."
        )
    # Greetings & Smalltalk
    elif any(x in intent for x in ["greeting", "hello", "hi", "thank", "bye", "goodbye", "appreciate", "smalltalk"]):
        intent_guidance = (
            "Respond naturally and warmly. "
            "Keep it brief and friendly. "
            "For greetings, invite them to share what they need help with. "
            "For thanks or goodbyes, wish them well genuinely."
        )
    # General / fallback guidance
    else:
        intent_guidance = (
            "Understand what the customer needs and respond helpfully. "
            "Be clear, specific and actionable in your response. "
            "If more information is needed, ask one focused question."
        )

    # ── Conversation context ────────────────────
    context = ""
    if history and len(history) > 0:
        context = "\nPrevious conversation:\n"
        for turn in history[-4:]:
            role     = "Customer" if turn["role"] == "user" else "Bot"
            context += f"  {role}: {turn['content']}\n"
        context += "\nContinue naturally from the above. Now respond to the latest query:\n"

    # ── Final prompt ────────────────────────────
    return (
        f"You are a professional customer service chatbot for a financial and services company.\n"
        f"{context}"
        f"Customer query: \"{query}\"\n\n"
        f"Topic detected: {intent_readable}\n\n"
        f"Tone instruction: {sentiment_tone}\n\n"
        f"How to handle this topic: {intent_guidance}\n\n"
        f"Response rules:\n"
        f"- Write 2-3 complete, helpful sentences\n"
        f"- Every sentence must be fully complete — never cut off mid-sentence\n"
        f"- Do not mention intent names, confidence scores or system labels\n"
        f"- Sound human, natural and genuinely helpful — not robotic\n"
        f"- End with a period or exclamation mark\n"
    )

# ─────────────────────────────────────────────
# RATE LIMIT HANDLER
# ─────────────────────────────────────────────
def wait_for_rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < REQUEST_DELAY:
        wait = REQUEST_DELAY - elapsed
        if wait > 1:
            print(f"  [Waiting {wait:.0f}s...]", flush=True)
        time.sleep(wait)
    _last_request_time = time.time()

# ─────────────────────────────────────────────
# PROVIDER API CALLS
# ─────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {ACTIVE_KEY}", "Content-Type": "application/json"}
    body    = {
        "model"      : ACTIVE_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : CFG["max_tokens"],
        "temperature": CFG["temperature"],
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()
    raise Exception(f"{r.status_code}: {r.json().get('error', {}).get('message', r.text[:100])}")


def call_gemini(prompt: str) -> str:
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_MODEL}:generateContent?key={ACTIVE_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": CFG["max_tokens"],
            "temperature"    : CFG["temperature"],
            "topP"           : 0.9,
        }
    }
    r = requests.post(url, json=body, timeout=30)
    if r.status_code == 200:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise Exception(f"{r.status_code}: {r.json().get('error', {}).get('message', r.text[:100])}")


def call_openai(prompt: str) -> str:
    url     = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {ACTIVE_KEY}", "Content-Type": "application/json"}
    body    = {
        "model"      : ACTIVE_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : CFG["max_tokens"],
        "temperature": CFG["temperature"],
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()
    raise Exception(f"{r.status_code}: {r.json().get('error', {}).get('message', r.text[:100])}")


def call_claude(prompt: str) -> str:
    url     = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key"        : ACTIVE_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type"     : "application/json",
    }
    body = {
        "model"     : ACTIVE_MODEL,
        "max_tokens": CFG["max_tokens"],
        "messages"  : [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code == 200:
        return r.json()["content"][0]["text"].strip()
    raise Exception(f"{r.status_code}: {r.json().get('error', {}).get('message', r.text[:100])}")


# ─────────────────────────────────────────────
# UNIFIED CALLER
# ─────────────────────────────────────────────
CALLERS = {
    "groq"  : call_groq,
    "gemini": call_gemini,
    "openai": call_openai,
    "claude": call_claude,
}

def get_response(query: str, intent: str, sentiment: str,
                 confidence: float, history: list = None) -> str:
    prompt = build_prompt(query, intent, sentiment, confidence, history)
    wait_for_rate_limit()

    for attempt in range(2):
        try:
            response = CALLERS[ACTIVE_PROVIDER](prompt)
            if response and response[-1] not in ".!?":
                response += "."
            return response

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt == 0:
                print(f"  [Rate limit — waiting 70s...]")
                time.sleep(70)
                _last_request_time = time.time()
                continue
            elif any(x in error_str for x in ["401", "403", "API_KEY", "api_key", "Invalid"]):
                return f"❌ API key error for {ACTIVE_PROVIDER.upper()}. Please check your key."
            else:
                print(f"  [Error: {error_str[:100]}]")
                return "I apologize, I am having trouble right now. Please try again."

    return "I apologize, the service is temporarily unavailable. Please try again shortly."

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def analyze(query: str, history: list = None) -> dict:
    t0       = time.time()
    bert_out = classify(query)
    t_ai     = time.time()
    response = get_response(
        query      = query,
        intent     = bert_out["intent"],
        sentiment  = bert_out["sentiment"],
        confidence = bert_out["intent_confidence"],
        history    = history,
    )
    return {
        "query"               : query,
        "intent"              : bert_out["intent"],
        "intent_confidence"   : bert_out["intent_confidence"],
        "top3_intents"        : bert_out["top3_intents"],
        "sentiment"           : bert_out["sentiment"],
        "sentiment_confidence": bert_out["sentiment_confidence"],
        "low_confidence"      : bert_out["low_confidence"],
        "pre_classified"      : bert_out["pre_classified"],
        "response"            : response,
        "bert_ms"             : round((t_ai - t0) * 1000, 1),
        "ai_ms"               : round((time.time() - t_ai) * 1000, 1),
        "latency_ms"          : round((time.time() - t0) * 1000, 1),
    }

# ─────────────────────────────────────────────
# SAMPLE TESTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  SAMPLE TESTS — using {ACTIVE_PROVIDER.upper()}")
print("=" * 60)

sample_queries = [
    "What is my current account balance?",
    "I am really frustrated my card has been blocked for no reason!",
    "Thank you so much you guys are absolutely amazing!",
    "someone else is using my net banking",          # ← security test
    "I didn't authorize this transaction on my card", # ← fraud test
]

for query in sample_queries:
    print(f"\n  {'─'*55}")
    result    = analyze(query)
    emoji     = SENTIMENT_EMOJI[result["sentiment"]]
    conf_flag = " ⚠️ LOW CONF"  if result["low_confidence"]  else ""
    safe_flag = " 🚨 SECURITY"  if result["pre_classified"]  else ""
    print(f"  Query     : {result['query']}")
    print(f"  Intent    : {result['intent']}  ({result['intent_confidence']*100:.1f}%){conf_flag}{safe_flag}")
    print(f"  Sentiment : {emoji} {result['sentiment']}  ({result['sentiment_confidence']*100:.1f}%)")
    print(f"  Response  : {result['response']}")
    print(f"  Latency   : BERT {result['bert_ms']}ms + {ACTIVE_PROVIDER.upper()} {result['ai_ms']}ms = {result['latency_ms']}ms")

# ─────────────────────────────────────────────
# LIVE CHATBOT DEMO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  LIVE CHATBOT — powered by {ACTIVE_PROVIDER.upper()}")
print("  BERT classifies → AI responds")
if ACTIVE_PROVIDER == "gemini":
    print("  Note: 20 second wait between responses (free tier)")
print("  ─────────────────────────────────────────────────")
print("  Try: 'My card is blocked and I am furious!'")
print("  Try: 'Thank you so much this is wonderful!'")
print("  Try: 'What is my account balance?'")
print("  Try: 'I need to book a flight to London'")
print("  Commands  : 'history' | 'clear' | 'quit'")
print("=" * 60)

conversation_history = []   # from pipeline.py — tracks last 4 exchanges
turn_log             = []   # from pipeline.py — for 'history' command

while True:
    try:
        print()
        query = input("  You: ").strip()

        if not query:
            print("  Please type something!")
            continue
        if query.lower() in ("quit", "exit", "q"):
            print(f"\n  Goodbye! Total turns: {len(turn_log)}")
            break

        # from pipeline.py — history command
        if query.lower() == "history":
            if not turn_log:
                print("  No conversation history yet.")
            else:
                print("\n  ── Conversation History ──")
                for i, t in enumerate(turn_log, 1):
                    print(f"  [{i}] You : {t['query']}")
                    print(f"       Bot : {t['response']}")
            continue

        # from pipeline.py — clear command
        if query.lower() == "clear":
            conversation_history = []
            turn_log             = []
            print("  Conversation cleared.")
            continue

        result    = analyze(query, conversation_history)
        emoji     = SENTIMENT_EMOJI[result["sentiment"]]
        conf_flag = " ⚠️ LOW CONF" if result["low_confidence"]  else ""
        safe_flag = " 🚨 SECURITY" if result["pre_classified"]  else ""

        # update history — keep last 4 exchanges
        conversation_history.append({"role": "user",  "content": query})
        conversation_history.append({"role": "model", "content": result["response"]})
        if len(conversation_history) > 8:
            conversation_history = conversation_history[-8:]

        turn_log.append({"query": query, "response": result["response"]})

        print(f"\n  Bot       : {result['response']}")
        print(f"  {'─'*50}")
        print(f"  Intent    : {result['intent']}  ({result['intent_confidence']*100:.1f}%){conf_flag}{safe_flag}")
        print(f"  Sentiment : {emoji} {result['sentiment']}  ({result['sentiment_confidence']*100:.1f}%)")
        print(f"  Top 3     : {[(n, f'{s}%') for n, s in result['top3_intents']]}")
        print(f"  Latency   : BERT {result['bert_ms']}ms + {ACTIVE_PROVIDER.upper()} {result['ai_ms']}ms = {result['latency_ms']}ms")
        if result["low_confidence"]:
            print(f"  ℹ️  Query was unclear — responded with fallback message")
        if result["pre_classified"]:
            print(f"  🚨 Security/fraud intent detected — urgent response triggered")

    except KeyboardInterrupt:
        print("\n  Exiting...")
        break
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE ✅")
print("=" * 60)
print(f"""
  BERT      → Intent Classification : 86.20% accuracy
  BERT      → Sentiment Analysis    : 93.13% accuracy
  {ACTIVE_PROVIDER.upper():<10} → Response Generation  : {ACTIVE_MODEL}
""")