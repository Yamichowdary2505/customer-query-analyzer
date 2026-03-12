"""
=============================================================
  CUSTOMER QUERY ANALYZER — STREAMLIT DASHBOARD
  AI-Based Customer Query Analyzer

  Features:
  - BERT multi-task: intent (151) + sentiment
  - Safety net for fraud/security queries
  - Conversation history + multi-turn context
  - Low confidence fallback
  - Multi-provider: Groq / Gemini / OpenAI / Claude
  - Real-time analytics panel

  Run locally : streamlit run app.py
  Deploy      : Push to GitHub → Streamlit Cloud
=============================================================
"""

import re
import json
import time
import requests
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title            = "Customer Query Analyzer",
    page_icon             = "🤖",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: #0f1628;
    border-right: 1px solid #1e2d4a;
    display: block !important;
    visibility: visible !important;
    min-width: 260px !important;
    transform: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #60a5fa; }

.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f1f3d 50%, #162032 100%);
    border: 1px solid #2a4a7f;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(96,165,250,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; color: #f0f9ff;
    margin: 0 0 6px 0; letter-spacing: -0.5px;
}
.main-header p { color: #94a3b8; margin: 0; font-size: 0.95rem; font-weight: 300; }
.header-badge {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.3);
    color: #60a5fa; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 8px; margin-bottom: 12px;
}
.metric-card {
    background: #0f1628; border: 1px solid #1e2d4a;
    border-radius: 12px; padding: 20px;
    text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3b82f6; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    color: #60a5fa; line-height: 1;
}
.metric-label {
    color: #64748b; font-size: 0.78rem;
    margin-top: 6px; text-transform: uppercase; letter-spacing: 1px;
}
.chat-container {
    background: #0f1628; border: 1px solid #1e2d4a;
    border-radius: 16px; padding: 20px;
    height: 440px; overflow-y: auto; margin-bottom: 16px;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white; padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 4px 20%; font-size: 0.92rem; line-height: 1.5;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
}
.chat-bubble-bot {
    background: #1a2540; border: 1px solid #2a3f6a;
    color: #e2e8f0; padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 4px 0; font-size: 0.92rem; line-height: 1.5;
}
.chat-bubble-security {
    background: #2d1a1a; border: 1px solid #7f1d1d;
    color: #fecaca; padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 4px 0; font-size: 0.92rem; line-height: 1.5;
}
.chat-meta {
    font-size: 0.72rem; color: #475569;
    margin-bottom: 8px; font-family: 'Space Mono', monospace;
}
.chat-meta-right { text-align: right; margin-right: 4px; }
.tag {
    display: inline-block; padding: 2px 8px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 500; margin: 2px;
}
.tag-intent  { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4); color: #a5b4fc; }
.tag-neg     { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.35); color: #fca5a5; }
.tag-neu     { background: rgba(100,116,139,0.15);border: 1px solid rgba(100,116,139,0.4);color: #94a3b8; }
.tag-pos     { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.35); color: #86efac; }
.tag-security{ background: rgba(239,68,68,0.2);   border: 1px solid rgba(239,68,68,0.5);  color: #fca5a5; }
.tag-lowconf { background: rgba(234,179,8,0.15);  border: 1px solid rgba(234,179,8,0.4);  color: #fde68a; }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.78rem;
    color: #60a5fa; text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1e2d4a;
}
/* ── All text inputs ── */
.stTextInput input, .stTextInput textarea,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background: #0f1628 !important;
    border: 1px solid #2a3f6a !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    caret-color: #60a5fa !important;
}
.stTextInput input:focus,
div[data-baseweb="input"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
    outline: none !important;
}
/* ── Placeholder text — must be visible on dark bg ── */
.stTextInput input::placeholder,
div[data-baseweb="input"] input::placeholder,
input::placeholder {
    color: #4a6080 !important;
    opacity: 1 !important;
}
/* ── Select boxes ── */
div[data-baseweb="select"] > div {
    background: #0f1628 !important;
    border: 1px solid #2a3f6a !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] svg { fill: #60a5fa !important; }
div[data-baseweb="popover"] {
    background: #0f1628 !important;
    border: 1px solid #2a3f6a !important;
}
li[role="option"] {
    background: #0f1628 !important;
    color: #e2e8f0 !important;
}
li[role="option"]:hover { background: #1e2d4a !important; }
/* ── Password input eye icon ── */
div[data-baseweb="input"] button { color: #60a5fa !important; }
/* ── Labels ── */
.stTextInput label, .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
/* ── Expander ── */
details summary {
    color: #60a5fa !important;
    font-size: 0.82rem !important;
}
details {
    background: #0f1628 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    padding: 4px 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 500 !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.88 !important; }
.conf-bar-bg  { background: #1e2d4a; border-radius: 6px; height: 8px; margin: 4px 0 12px 0; }
.conf-bar-fill{ height: 8px; border-radius: 6px; background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.conf-bar-sec { height: 8px; border-radius: 6px; background: linear-gradient(90deg, #ef4444, #f87171); }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2a3f6a; border-radius: 3px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "messages"         : [],
    "conv_history"     : [],
    "history_log"      : [],
    "total_queries"    : 0,
    "sentiment_counts" : {"negative": 0, "neutral": 0, "positive": 0},
    "security_count"   : 0,
    "lowconf_count"    : 0,
    "bert_loaded"      : False,
    "last_result"      : None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, dict) else v.copy()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SENTIMENT_EMOJI    = {"negative": "😠", "neutral": "😐", "positive": "😊"}
SENTIMENT_COLOR    = {"negative": "tag-neg", "neutral": "tag-neu", "positive": "tag-pos"}
SENTIMENT_NAMES    = ["negative", "neutral", "positive"]
LOW_CONF_THRESHOLD = 0.20

MODELS = {
    "groq"  : "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
}

# ─────────────────────────────────────────────
# SAFETY NET
# ─────────────────────────────────────────────
SAFETY_PATTERNS = {
    "unauthorized_access": [
        "someone else", "someone is using", "unauthori", "hacked", "hack",
        "not me", "wasn't me", "i didn't do", "suspicious login", "unknown login",
        "someone logged", "someone accessed", "strange activity", "unusual activity",
        "unknown transaction", "i didn't make this", "i did not make", "fraudulent login",
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
    q = query.lower()
    for intent, keywords in SAFETY_PATTERNS.items():
        for kw in keywords:
            if kw in q:
                return intent, 0.95
    return None, None

# ─────────────────────────────────────────────
# BERT MODEL
# ─────────────────────────────────────────────
class MultiTaskBERT(nn.Module):
    def __init__(self, bert_model_name, num_intents, num_sentiments, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_intents)
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_sentiments)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = self.dropout(out.pooler_output)
        return self.intent_classifier(cls), self.sentiment_classifier(cls)

@st.cache_resource(show_spinner=False)
def load_model(model_dir, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{data_dir}/intent_label_map.json") as f:
        id2intent = json.load(f)
    num_intents = len(id2intent)
    oos_id      = next((int(k) for k, v in id2intent.items() if v == "oos"), -1)
    tokenizer   = BertTokenizer.from_pretrained(model_dir)
    model       = MultiTaskBERT("bert-base-uncased", num_intents, 3)
    model.load_state_dict(
        torch.load(f"{model_dir}/bert_best.pt", map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, id2intent, oos_id, device

def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\'\-\?\!\.,]", "", text)
    text = re.sub(r"(\w)\1{3,}", r"\1\1", text)
    return text

@torch.no_grad()
def classify(query, model, tokenizer, id2intent, oos_id, device):
    clean_query = clean_text(query)
    override_intent, override_conf = pre_classify(clean_query)

    enc = tokenizer(clean_query, max_length=64, padding="max_length",
                    truncation=True, return_tensors="pt")
    il, sl = model(enc["input_ids"].to(device),
                   enc["attention_mask"].to(device),
                   enc["token_type_ids"].to(device))
    ip = torch.softmax(il, dim=-1)[0]
    sp = torch.softmax(sl, dim=-1)[0]
    iid         = ip.argmax().item()
    sid         = sp.argmax().item()
    intent_conf = ip[iid].item()

    if override_intent:
        intent_name    = override_intent
        intent_conf    = override_conf
        low_conf       = False
        pre_classified = True
    elif intent_conf < LOW_CONF_THRESHOLD and oos_id >= 0:
        intent_name    = "out_of_scope"
        low_conf       = True
        pre_classified = False
    else:
        intent_name    = id2intent[str(iid)]
        low_conf       = False
        pre_classified = False

    top3_idx    = ip.topk(3).indices.cpu().numpy()
    top3_scores = ip.topk(3).values.cpu().numpy()

    return {
        "intent"              : intent_name,
        "intent_confidence"   : round(intent_conf, 4),
        "top3_intents"        : [(id2intent[str(i)], round(float(s)*100, 1))
                                  for i, s in zip(top3_idx, top3_scores)],
        "sentiment"           : SENTIMENT_NAMES[sid],
        "sentiment_confidence": round(sp[sid].item(), 4),
        "sentiment_scores"    : {
            "negative": round(sp[0].item()*100, 1),
            "neutral" : round(sp[1].item()*100, 1),
            "positive": round(sp[2].item()*100, 1),
        },
        "low_confidence"  : low_conf,
        "pre_classified"  : pre_classified,
    }

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
def build_prompt(query, intent, sentiment, confidence, history=None):
    if intent in ("oos", "out_of_scope") or confidence < LOW_CONF_THRESHOLD:
        if history and len(history) > 0:
            context = "\nPrevious conversation:\n"
            for turn in history[-4:]:
                role = "Customer" if turn["role"] == "user" else "Bot"
                context += f"  {role}: {turn['content']}\n"
            return (
                f"You are a professional customer service chatbot.\n{context}\n"
                f"Customer's latest message: \"{query}\"\n\n"
                f"Instructions:\n"
                f"- Message is short or unclear on its own\n"
                f"- Use conversation history to understand what they mean\n"
                f"- Respond naturally as if continuing the conversation\n"
                f"- Do NOT ask to rephrase if context makes it clear\n"
                f"- Write 2-3 complete helpful sentences\n"
                f"- Never mention intent names or confidence scores"
            )
        return (
            f"You are a helpful customer service chatbot.\n"
            f"Customer said: \"{query}\"\n"
            f"You could not confidently understand this request.\n"
            f"- Apologize briefly and warmly\n"
            f"- Ask to rephrase or provide more details\n"
            f"- Suggest topics: account, payments, cards, orders, bookings\n"
            f"- Write 2-3 complete sentences\n"
            f"- Never mention confidence scores or intent labels"
        )

    intent_readable = intent.replace("_", " ")
    sentiment_tone  = {
        "negative": "Customer is frustrated. Open with genuine apology. Be calm, reassuring, solution-focused. Never blame the customer.",
        "neutral" : "Customer making a calm request. Be professional, clear, concise. Provide accurate actionable information.",
        "positive": "Customer is happy. Match their positive energy with warmth. Be friendly and personable. Make them feel valued.",
    }.get(sentiment, "Be professional, helpful and polite.")

    intent_guidance = ""
    if intent == "unauthorized_access":
        intent_guidance = "URGENT: Advise: 1) Change password now, 2) Enable 2FA, 3) Review recent logins, 4) Contact security team."
    elif intent == "report_fraud":
        intent_guidance = "URGENT: Advise: 1) Block card, 2) File dispute, 3) Note transaction details. Reassure they are protected."
    elif intent == "emergency_block":
        intent_guidance = "URGENT: Block card immediately via app or helpline. Offer replacement card."
    elif intent == "account_compromised":
        intent_guidance = "URGENT: Reset password immediately. Check if registered email/phone was changed. Contact support if locked out."
    elif any(x in intent for x in ["balance", "account", "bank", "statement"]):
        intent_guidance = "Guide to check via app/website. Remind to keep credentials secure."
    elif any(x in intent for x in ["card", "freeze", "block", "pin", "chip"]):
        intent_guidance = "Take card issues seriously. Provide clear next steps for blocking, replacing or unlocking."
    elif any(x in intent for x in ["transfer", "transaction", "payment", "send", "wire"]):
        intent_guidance = "Be precise. Confirm transaction type. Provide processing times and fees."
    elif any(x in intent for x in ["fraud", "dispute", "unauthorized", "stolen", "scam"]):
        intent_guidance = "Highest urgency. Reassure customer they are protected. Guide to report/dispute."
    elif any(x in intent for x in ["order", "delivery", "shipping", "track", "return", "refund"]):
        intent_guidance = "Acknowledge concern. Provide tracking steps or resolution timeline."
    elif any(x in intent for x in ["book", "flight", "hotel", "reservation", "ticket", "travel"]):
        intent_guidance = "Help with booking clearly. Confirm details. Provide next steps."
    elif any(x in intent for x in ["loan", "credit", "mortgage", "interest", "borrow"]):
        intent_guidance = "Be transparent. Use 'typically' or 'generally'. Avoid making promises."
    elif any(x in intent for x in ["app", "login", "password", "reset", "otp", "verify"]):
        intent_guidance = "Guide through steps clearly. Reassure issue is common and easily resolved."
    elif any(x in intent for x in ["greeting", "hello", "thank", "bye", "goodbye"]):
        intent_guidance = "Respond naturally and warmly. Keep brief. Invite them to share their need."
    else:
        intent_guidance = "Understand the need and respond helpfully. Be clear and actionable."

    context = ""
    if history and len(history) > 0:
        context = "\nPrevious conversation:\n"
        for turn in history[-4:]:
            role = "Customer" if turn["role"] == "user" else "Bot"
            context += f"  {role}: {turn['content']}\n"
        context += "\nContinue naturally. Now respond to the latest query:\n"

    return (
        f"You are a professional customer service chatbot for a financial and services company.\n"
        f"{context}"
        f"Customer query: \"{query}\"\n\n"
        f"Topic detected: {intent_readable}\n\n"
        f"Tone instruction: {sentiment_tone}\n\n"
        f"How to handle this topic: {intent_guidance}\n\n"
        f"Response rules:\n"
        f"- Write 2-3 complete, helpful sentences\n"
        f"- Every sentence must be fully complete\n"
        f"- Do not mention intent names, confidence scores or system labels\n"
        f"- Sound human, natural and genuinely helpful\n"
        f"- End with a period or exclamation mark\n"
    )

# ─────────────────────────────────────────────
# AI RESPONSE
# ─────────────────────────────────────────────
def get_ai_response(query, intent, sentiment, confidence, provider, api_key, history=None):
    prompt = build_prompt(query, intent, sentiment, confidence, history)
    model  = MODELS[provider]
    try:
        if provider == "groq":
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200, "temperature": 0.7}, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7}}, timeout=30)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "openai":
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200, "temperature": 0.7}, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

        elif provider == "claude":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 200,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=30)
            if r.status_code == 200:
                return r.json()["content"][0]["text"].strip()

        return f"API Error {r.status_code} — please check your key."
    except Exception as e:
        return f"Connection error: {str(e)[:80]}"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px 0;'>
        <div style='font-size:2rem;'>🤖</div>
        <div style='font-family:Space Mono,monospace;font-size:0.9rem;color:#60a5fa;font-weight:700;margin-top:4px;letter-spacing:1px;'>QUERY ANALYZER</div>
        <div style='font-size:0.72rem;color:#334155;margin-top:2px;'>BERT + LLM Pipeline</div>
    </div>
    <div style='border-bottom:1px solid #1e2d4a;margin-bottom:16px;'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.78rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>🔌 AI Provider</div>", unsafe_allow_html=True)
    provider = st.selectbox("AI Provider", ["groq", "gemini", "openai", "claude"],
                            index=0, label_visibility="collapsed",
                            help="Groq is free and recommended")



    # API key — auto from secrets on cloud, manual input locally
    api_key = ""
    try:
        import streamlit.runtime.secrets as _sec
        _store = _sec.SecretsManager()
        secret_map = {"groq":"GROQ_API_KEY","gemini":"GEMINI_API_KEY","openai":"OPENAI_API_KEY","claude":"CLAUDE_API_KEY"}
        api_key = _store[secret_map[provider]]
    except Exception:
        api_key = ""

    if api_key:
        st.markdown("<div style='font-size:0.75rem;color:#22c55e;margin:6px 0 12px 0;'>✅ API key loaded securely from secrets</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.78rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:12px 0 8px 0;'>🔑 API Key</div>", unsafe_allow_html=True)
        api_key = st.text_input(
            f"{provider.upper()} API Key",
            type="password",
            placeholder="Paste your API key here...",
            help={"groq":"FREE — get at console.groq.com","gemini":"Free tier — aistudio.google.com",
                  "openai":"Paid — platform.openai.com","claude":"Paid — console.anthropic.com"}[provider]
        )
        if api_key:
            masked = api_key[:4] + "•" * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "•" * len(api_key)
            st.markdown(f"<div style='font-size:0.75rem;color:#22c55e;margin:-8px 0 8px 0;'>✅ Key entered: {masked}</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Advanced: Model Paths", expanded=False):
        model_path = st.text_input(
            "BERT Model Path",
            value=r"C:\project_s",
            help="Folder containing bert_best.pt and tokenizer files",
            placeholder=r"e.g. C:\project_s"
        )
        data_path = st.text_input(
            "Data Path",
            value=r"C:\project_s\clinc_oos\pre_processed",
            help="Folder containing intent_label_map.json",
            placeholder=r"e.g. C:\project_s\clinc_oos\pre_processed"
        )
        st.markdown("<div style='font-size:0.72rem;color:#475569;margin-top:4px;'>These paths are only used locally and never sent anywhere.</div>", unsafe_allow_html=True)

    load_btn = st.button("🚀 Load Model", use_container_width=True)

    st.markdown("<div style='border-bottom:1px solid #1e2d4a;margin:16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>📊 Session Stats</div>", unsafe_allow_html=True)
    total = st.session_state.total_queries
    neg   = st.session_state.sentiment_counts["negative"]
    neu   = st.session_state.sentiment_counts["neutral"]
    pos   = st.session_state.sentiment_counts["positive"]
    sec   = st.session_state.security_count
    low   = st.session_state.lowconf_count

    st.markdown(f"""
    <div style='font-size:0.85rem;color:#94a3b8;'>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span>Total Queries</span><span style='color:#60a5fa;font-weight:600;'>{total}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span>😠 Negative</span><span style='color:#fca5a5;'>{neg}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span>😐 Neutral</span><span style='color:#94a3b8;'>{neu}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span>😊 Positive</span><span style='color:#86efac;'>{pos}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid #1e2d4a;margin-top:4px;padding-top:8px;'>
            <span>🚨 Security Alerts</span><span style='color:#fca5a5;'>{sec}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span>⚠️ Low Confidence</span><span style='color:#fde68a;'>{low}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom:1px solid #1e2d4a;margin:16px 0;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v if not isinstance(v, dict) else v.copy()
        st.rerun()

    st.markdown("<div style='border-bottom:1px solid #1e2d4a;margin:16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#475569;'>
        <b style='color:#60a5fa;'>Model Performance</b><br>
        Intent Accuracy : <b style='color:#86efac;'>86.20%</b><br>
        Sentiment Accuracy : <b style='color:#86efac;'>93.13%</b><br>
        Dataset : CLINC150 (151 intents)<br>
        Architecture : BERT Multi-task<br><br>
        <b style='color:#60a5fa;'>Safety Net</b><br>
        Unauthorized Access · Fraud<br>
        Emergency Block · Compromised Account
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <span class="header-badge">BERT</span>
        <span class="header-badge">Multi-Task NLP</span>
        <span class="header-badge">151 Intents</span>
        <span class="header-badge">Safety Net</span>
    </div>
    <h1>🤖 Customer Query Analyzer</h1>
    <p>AI-powered intent classification · sentiment analysis · intelligent response generation</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
if load_btn:
    if not api_key:
        st.warning("Please paste your API key in the sidebar to continue.")
    else:
        with st.spinner("Loading BERT model... (~10 seconds on first load)"):
            try:
                model, tokenizer, id2intent, oos_id, device = load_model(model_path, data_path)
                st.session_state.update({
                    "bert_loaded": True, "model": model, "tokenizer": tokenizer,
                    "id2intent": id2intent, "oos_id": oos_id, "device": device,
                    "provider": provider, "api_key": api_key,
                })
                st.success(f"✅ BERT loaded! {len(id2intent)} intents · {provider.upper()} ({MODELS[provider]})")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.info("Check model path and data path in the sidebar.")

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_chat, col_analytics = st.columns([1.1, 0.9], gap="large")

# ══════════════════════════════════════════════
# LEFT — CHAT
# ══════════════════════════════════════════════
with col_chat:
    st.markdown('<div class="section-header">💬 Chat Interface</div>', unsafe_allow_html=True)

    chat_html = '<div class="chat-container">'
    if not st.session_state.messages:
        chat_html += """
        <div style='text-align:center;color:#334155;margin-top:150px;'>
            <div style='font-size:2.5rem;'>🤖</div>
            <div style='font-size:0.9rem;margin-top:8px;'>Load the model and start chatting</div>
            <div style='font-size:0.78rem;margin-top:6px;color:#1e3a5f;'>
                Try: "What is my balance?" · "Someone hacked my account!" · "I lost my card"
            </div>
        </div>"""
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-bubble-user">{msg['content']}</div>
                <div class="chat-meta chat-meta-right">{msg['time']}</div>"""
            else:
                is_sec    = msg.get("pre_classified", False)
                is_low    = msg.get("low_confidence", False)
                bubble    = "chat-bubble-security" if is_sec else "chat-bubble-bot"
                i_label   = msg.get("intent","").replace("_"," ")
                i_tag     = f'<span class="tag tag-intent">🎯 {i_label}</span>'
                s_cls     = SENTIMENT_COLOR.get(msg.get("sentiment","neutral"),"tag-neu")
                s_emoji   = SENTIMENT_EMOJI.get(msg.get("sentiment","neutral"),"😐")
                s_tag     = f'<span class="tag {s_cls}">{s_emoji} {msg.get("sentiment","")}</span>'
                extra     = ""
                if is_sec: extra += '<span class="tag tag-security">🚨 SECURITY</span>'
                if is_low: extra += '<span class="tag tag-lowconf">⚠️ LOW CONF</span>'
                chat_html += f"""
                <div class="{bubble}">{msg['content']}</div>
                <div class="chat-meta">{i_tag} {s_tag} {extra} · {msg['time']} · {msg.get('latency','')}</div>"""
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "q", label_visibility="collapsed",
                placeholder="💬  Type your query here...  e.g. 'What is my balance?' or 'Someone hacked my account'"
            )
        with c2:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input:
        if not st.session_state.bert_loaded:
            st.warning("⚠️ Please load the model first using the sidebar button.")
        else:
            with st.spinner("Analyzing..."):
                t0     = time.time()
                result = classify(user_input, st.session_state.model,
                                  st.session_state.tokenizer, st.session_state.id2intent,
                                  st.session_state.oos_id, st.session_state.device)
                response = get_ai_response(
                    user_input, result["intent"], result["sentiment"],
                    result["intent_confidence"], st.session_state.provider,
                    st.session_state.api_key, st.session_state.conv_history)
                latency = round((time.time() - t0) * 1000)
                now     = datetime.now().strftime("%H:%M")

            # Update conv history
            st.session_state.conv_history.append({"role": "user",  "content": user_input})
            st.session_state.conv_history.append({"role": "model", "content": response})
            if len(st.session_state.conv_history) > 8:
                st.session_state.conv_history = st.session_state.conv_history[-8:]

            st.session_state.messages.append({"role": "user", "content": user_input, "time": now})
            st.session_state.messages.append({
                "role": "bot", "content": response,
                "intent": result["intent"], "sentiment": result["sentiment"],
                "pre_classified": result["pre_classified"],
                "low_confidence": result["low_confidence"],
                "time": now, "latency": f"{latency}ms",
            })

            st.session_state.total_queries += 1
            st.session_state.sentiment_counts[result["sentiment"]] += 1
            if result["pre_classified"]: st.session_state.security_count += 1
            if result["low_confidence"]: st.session_state.lowconf_count  += 1

            st.session_state.last_result = {**result, "response": response,
                                             "latency": latency, "query": user_input}

            flag = "🚨" if result["pre_classified"] else ("⚠️" if result["low_confidence"] else "✅")
            st.session_state.history_log.append({
                "Time"      : now,
                "Query"     : user_input[:45] + "..." if len(user_input) > 45 else user_input,
                "Intent"    : result["intent"].replace("_", " "),
                "Confidence": f"{result['intent_confidence']*100:.1f}%",
                "Sentiment" : f"{SENTIMENT_EMOJI[result['sentiment']]} {result['sentiment']}",
                "Status"    : flag,
                "Latency"   : f"{latency}ms",
            })
            st.rerun()

# ══════════════════════════════════════════════
# RIGHT — ANALYTICS
# ══════════════════════════════════════════════
with col_analytics:
    st.markdown('<div class="section-header">📊 Analytics Panel</div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        r = st.session_state.last_result
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:0.9rem;line-height:1.4;">{r['intent'].replace('_',' ')}</div>
                <div class="metric-label">Intent</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{SENTIMENT_EMOJI[r['sentiment']]}</div>
                <div class="metric-label">{r['sentiment'].capitalize()}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            flag_d = "🚨" if r["pre_classified"] else ("⚠️" if r["low_confidence"] else "✅")
            label  = "Security" if r["pre_classified"] else ("Low Conf" if r["low_confidence"] else "Normal")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">{flag_d}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">🎯 Intent Confidence (Top 3)</div>', unsafe_allow_html=True)
        for name, score in r["top3_intents"]:
            bar = "conf-bar-sec" if r["pre_classified"] else "conf-bar-fill"
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;font-size:0.82rem;color:#94a3b8;margin-bottom:4px;'>
                    <span>{name.replace('_',' ')}</span>
                    <span style='color:#60a5fa;font-family:Space Mono,monospace;'>{score}%</span>
                </div>
                <div class="conf-bar-bg"><div class="{bar}" style="width:{min(score,100)}%;"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">💭 Sentiment Breakdown</div>', unsafe_allow_html=True)
        ss  = r["sentiment_scores"]
        fig = go.Figure(go.Bar(
            x=list(ss.values()), y=["Negative","Neutral","Positive"],
            orientation="h", marker_color=["#ef4444","#64748b","#22c55e"],
            text=[f"{v}%" for v in ss.values()], textposition="auto",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", height=160, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0,100]),
            yaxis=dict(showgrid=False), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center;color:#334155;padding:60px 20px;
                    background:#0f1628;border:1px solid #1e2d4a;border-radius:12px;'>
            <div style='font-size:2rem;'>📊</div>
            <div style='margin-top:8px;font-size:0.9rem;'>Analytics will appear after your first query</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.total_queries > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Session Overview</div>', unsafe_allow_html=True)
        counts = st.session_state.sentiment_counts
        fig2   = go.Figure(go.Pie(
            labels=["Negative","Neutral","Positive"],
            values=[counts["negative"], counts["neutral"], counts["positive"]],
            hole=0.6, marker_colors=["#ef4444","#64748b","#22c55e"],
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
            height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            annotations=[dict(text=f"<b>{st.session_state.total_queries}</b><br>queries",
                              x=0.5, y=0.5, font_size=14, font_color="#60a5fa", showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# QUERY HISTORY TABLE
# ─────────────────────────────────────────────
if st.session_state.history_log:
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Query History</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.history_log)
    st.dataframe(df, use_container_width=True, hide_index=True,
        column_config={
            "Query"     : st.column_config.TextColumn("Query", width="large"),
            "Intent"    : st.column_config.TextColumn("Intent"),
            "Confidence": st.column_config.TextColumn("Conf"),
            "Sentiment" : st.column_config.TextColumn("Sentiment"),
            "Status"    : st.column_config.TextColumn("Status"),
            "Latency"   : st.column_config.TextColumn("Latency"),
        })