"""
Customer Query Analyzer — Streamlit App
Final Year Project

Run: streamlit run app.py
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

st.set_page_config(
    page_title="Customer Query Analyzer",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #f5f7fa; }
.block-container { padding: 1.4rem 2rem 2rem 2rem !important; max-width: 100% !important; }
.main .block-container { max-width: 100% !important; width: 100% !important; padding: 1.4rem 2rem 2rem 2rem !important; }
section.main { max-width: 100% !important; }
[data-testid="stAppViewContainer"] > section.main { padding-left: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dde1e7;
    min-width: 240px;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1rem !important;
}

/* ── Sidebar toggle button (open/close arrow) ── */
[data-testid="stSidebarCollapseButton"] {
    background: #1a3c5e !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="stSidebarCollapseButton"]:hover {
    background: #14304e !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: #1a3c5e !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #14304e !important;
}

/* When sidebar is collapsed — the expand button floats on the main page */
[data-testid="collapsedControl"] {
    background: #1a3c5e !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="collapsedControl"]:hover {
    background: #14304e !important;
}
[data-testid="collapsedControl"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
[data-testid="collapsedControl"] button {
    background: #1a3c5e !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="collapsedControl"] button:hover {
    background: #14304e !important;
}

/* ── Page header ── */
.page-header {
    background: #1a3c5e;
    border-radius: 10px;
    padding: 24px 30px;
    margin-bottom: 18px;
    color: #ffffff;
}
.page-header h1 {
    font-size: 1.55rem;
    font-weight: 600;
    margin: 0 0 5px 0;
    letter-spacing: -0.2px;
    color: #ffffff;
}
.page-header p {
    margin: 0;
    font-size: 0.84rem;
    color: rgba(255,255,255,0.7);
    font-weight: 300;
}
.header-tags {
    margin-bottom: 10px;
}
.htag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    color: rgba(255,255,255,0.9);
    padding: 2px 10px;
    border-radius: 3px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 5px;
    letter-spacing: 0.3px;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b7280;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e9ecef;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar section labels ── */
.sb-sec {
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #9ca3af;
    margin: 14px 0 6px 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chat window ── */
.chat-window {
    background: #ffffff;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    padding: 14px 16px;
    height: 400px;
    overflow-y: auto;
    margin-bottom: 8px;
}
.bubble-user {
    background: #1a3c5e;
    color: #ffffff;
    padding: 9px 14px;
    border-radius: 14px 14px 3px 14px;
    margin: 5px 0 2px 18%;
    font-size: 0.86rem;
    line-height: 1.5;
}
.bubble-bot {
    background: #f8f9fb;
    border: 1px solid #dde1e7;
    color: #1f2937;
    padding: 9px 14px;
    border-radius: 14px 14px 14px 3px;
    margin: 5px 18% 2px 0;
    font-size: 0.86rem;
    line-height: 1.5;
}
.bubble-security {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    color: #7f1d1d;
    padding: 9px 14px;
    border-radius: 14px 14px 14px 3px;
    margin: 5px 18% 2px 0;
    font-size: 0.86rem;
    line-height: 1.5;
}
.msg-meta {
    font-size: 0.64rem;
    color: #9ca3af;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    align-items: center;
}

/* ── Tags ── */
.tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 0.64rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.t-intent { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.t-neg    { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
.t-neu    { background: #f9fafb; color: #374151; border: 1px solid #d1d5db; }
.t-pos    { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.t-sec    { background: #fef2f2; color: #b91c1c; border: 1px solid #f87171; font-weight: 700; }
.t-low    { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
.t-good   { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.t-bad    { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }

/* ── Confidence bars ── */
.bar-track { background: #e5e7eb; border-radius: 3px; height: 6px; margin: 3px 0 9px 0; overflow: hidden; }
.bar-blue  { background: #1a3c5e; height: 6px; border-radius: 3px; }
.bar-red   { background: #dc2626; height: 6px; border-radius: 3px; }

/* ── Metric tiles ── */
.metric-tile {
    background: #ffffff;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    padding: 12px 10px;
    text-align: center;
}
.metric-tile .val {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a3c5e;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-tile .lbl {
    font-size: 0.62rem;
    color: #9ca3af;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    color: #9ca3af;
    padding: 80px 20px;
}
.empty-state .text { font-size: 0.87rem; }
.empty-state .hint { font-size: 0.75rem; color: #d1d5db; margin-top: 5px; }

/* ── All buttons — clean outline style ── */
.stButton > button,
.stDownloadButton > button {
    background: #ffffff !important;
    color: #1a3c5e !important;
    border: 1.5px solid #1a3c5e !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1a3c5e !important;
    color: #ffffff !important;
}

/* Arrow submit button — square icon style, same height as input */
.stFormSubmitButton > button {
    background: #1a3c5e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 1.2rem !important;
    font-weight: 400 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 38px !important;
    line-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
.stFormSubmitButton > button:hover {
    background: #14304e !important;
    color: #ffffff !important;
}

/* Sidebar action buttons — filled */
section[data-testid="stSidebar"] .stButton > button {
    background: #1a3c5e !important;
    color: #ffffff !important;
    border: none !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #14304e !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: #ffffff !important;
    color: #1a3c5e !important;
    border: 1.5px solid #1a3c5e !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #1a3c5e !important;
    color: #ffffff !important;
}

/* ── Text inputs ── */
div[data-baseweb="input"] input,
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    caret-color: #1a3c5e !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}
div[data-baseweb="input"] input:focus {
    border-color: #1a3c5e !important;
    box-shadow: 0 0 0 2px rgba(26,60,94,0.1) !important;
    outline: none !important;
}

/* ── Selectbox — FIX for invisible dropdown text ── */
div[data-baseweb="select"] {
    background: #ffffff !important;
}
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
    min-height: 38px !important;
}
div[data-baseweb="select"] > div > div {
    color: #111827 !important;
}
/* The selected value text */
div[data-baseweb="select"] span {
    color: #111827 !important;
}
/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: #374151 !important;
    color: #374151 !important;
}
/* Dropdown popover */
div[data-baseweb="popover"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
/* Dropdown options */
ul[data-baseweb="menu"] li {
    background: #ffffff !important;
    color: #111827 !important;
    font-size: 0.86rem !important;
    font-family: 'Inter', sans-serif !important;
}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}

/* ── Password input icon ── */
div[data-baseweb="input"] button {
    color: #6b7280 !important;
}

/* ── Labels ── */
.stTextInput label,
.stSelectbox label {
    color: #374151 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Expander ── */
details {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
}
details summary {
    color: #1a3c5e !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f5f7fa; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container > div:first-child { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── JS: force main section to fill full width when sidebar collapses ──
# Streamlit sets an inline margin-left on section.main which CSS !important
# cannot override. We use a MutationObserver to remove it whenever it changes.
st.markdown("""
<script>
(function() {
    function fixWidth() {
        var main = document.querySelector('section.main');
        if (main) {
            main.style.setProperty('margin-left', '0px', 'important');
            main.style.setProperty('width', '100%', 'important');
            main.style.setProperty('max-width', '100%', 'important');
        }
    }
    fixWidth();
    var observer = new MutationObserver(fixWidth);
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });
    window.addEventListener('resize', fixWidth);
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
_defaults = {
    "messages"        : [],
    "conv_history"    : [],
    "history_log"     : [],
    "total_queries"   : 0,
    "sentiment_counts": {"negative": 0, "neutral": 0, "positive": 0},
    "security_count"  : 0,
    "lowconf_count"   : 0,
    "bert_loaded"     : False,
    "last_result"     : None,
    "intent_freq"     : {},
    "latencies"       : [],
    "feedback"        : {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        if isinstance(v, dict): st.session_state[k] = v.copy()
        elif isinstance(v, list): st.session_state[k] = []
        else: st.session_state[k] = v

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SENTIMENT_NAMES = ["negative", "neutral", "positive"]
SENTIMENT_LABEL = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
LOW_CONF        = 0.20

MODELS = {
    "groq"  : "llama-3.1-8b-instant",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
}

# ─────────────────────────────────────────────
# SAFETY NET
# ─────────────────────────────────────────────
SAFETY_PATTERNS = {
    "unauthorized_access": [
        "someone else","someone is using","unauthori","hacked","hack",
        "not me","wasn't me","i didn't do","suspicious login","unknown login",
        "someone logged","someone accessed","strange activity","unusual activity",
        "unknown transaction","i didn't make this","i did not make","fraudulent login",
    ],
    "report_fraud": [
        "fraud","scam","scammed","cheated","stolen","stole","theft",
        "fake transaction","unauthorized transaction","didn't authorize",
        "did not authorize","money missing","money gone","money disappeared",
        "deducted without","charged without","debited without my",
    ],
    "emergency_block": [
        "block immediately","block my card now","freeze immediately",
        "lost my card","card stolen","stolen card","i lost my",
        "cant find my card","missing card","card is missing",
    ],
    "account_compromised": [
        "account compromised","account breached","password changed",
        "someone changed my password","locked out","cant access my account",
        "cant log in","cant login","login not working","otp not received",
        "not receiving otp","verification not working",
    ],
}

def pre_classify(query: str):
    q = query.lower()
    for intent, kws in SAFETY_PATTERNS.items():
        for kw in kws:
            if kw in q:
                return intent, 0.95
    return None, None

# ─────────────────────────────────────────────
# BERT MODEL
# ─────────────────────────────────────────────
class MultiTaskBERT(nn.Module):
    def __init__(self, bert_name, num_intents, num_sentiments, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_name)
        h            = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, num_intents)
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_sentiments)
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
    n       = len(id2intent)
    oos_id  = next((int(k) for k, v in id2intent.items() if v == "oos"), -1)
    tok     = BertTokenizer.from_pretrained(model_dir)
    mdl     = MultiTaskBERT("bert-base-uncased", n, 3)
    mdl.load_state_dict(torch.load(f"{model_dir}/bert_best.pt", map_location=device, weights_only=True))
    mdl.to(device).eval()
    return mdl, tok, id2intent, oos_id, device


def clean_text(t):
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\'\-\?\!\.,]", "", t)
    t = re.sub(r"(\w)\1{3,}", r"\1\1", t)
    return t


@torch.no_grad()
def classify(query, mdl, tok, id2intent, oos_id, device):
    cq = clean_text(query)
    oi, oc = pre_classify(cq)
    enc = tok(cq, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    il, sl = mdl(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                 enc["token_type_ids"].to(device))
    ip = torch.softmax(il, dim=-1)[0]
    sp = torch.softmax(sl, dim=-1)[0]
    iid  = ip.argmax().item()
    sid  = sp.argmax().item()
    conf = ip[iid].item()

    if oi:
        intent_name = oi; conf = oc; low = False; pre = True
    elif conf < LOW_CONF and oos_id >= 0:
        intent_name = "out_of_scope"; low = True; pre = False
    else:
        intent_name = id2intent[str(iid)]; low = False; pre = False

    t3i = ip.topk(3).indices.cpu().numpy()
    t3s = ip.topk(3).values.cpu().numpy()
    return {
        "intent"              : intent_name,
        "intent_confidence"   : round(conf, 4),
        "top3_intents"        : [(id2intent[str(i)], round(float(s)*100, 1)) for i, s in zip(t3i, t3s)],
        "sentiment"           : SENTIMENT_NAMES[sid],
        "sentiment_confidence": round(sp[sid].item(), 4),
        "sentiment_scores"    : {
            "negative": round(sp[0].item()*100, 1),
            "neutral" : round(sp[1].item()*100, 1),
            "positive": round(sp[2].item()*100, 1),
        },
        "low_confidence": low,
        "pre_classified": pre,
    }

# ─────────────────────────────────────────────
# PROMPT BUILDER — unchanged
# ─────────────────────────────────────────────
def build_prompt(query, intent, sentiment, confidence, history=None):
    if intent in ("oos", "out_of_scope") or confidence < LOW_CONF:
        if history:
            ctx = "\nPrevious conversation:\n" + "".join(
                f"  {'Customer' if t['role']=='user' else 'Bot'}: {t['content']}\n"
                for t in history[-4:]
            )
            return (
                f"You are a professional customer service chatbot.\n{ctx}\n"
                f"Customer's latest message: \"{query}\"\n\n"
                f"Use conversation history to understand context. Respond naturally. "
                f"Write 2-3 complete helpful sentences. Never mention intent names or confidence scores."
            )
        return (
            f"You are a helpful customer service chatbot.\nCustomer said: \"{query}\"\n"
            f"You could not confidently understand this request.\n"
            f"Apologize briefly. Ask to rephrase. Suggest topics: account, payments, cards, orders, bookings. "
            f"Write 2-3 complete sentences. Never mention confidence scores or intent labels."
        )

    ir = intent.replace("_", " ")
    tone = {
        "negative": "Customer is frustrated. Open with genuine apology. Be calm, reassuring, solution-focused.",
        "neutral" : "Customer making a calm request. Be professional, clear, concise.",
        "positive": "Customer is happy. Match positive energy with warmth.",
    }.get(sentiment, "Be professional, helpful and polite.")

    if intent == "unauthorized_access":
        guide = "URGENT: Advise: 1) Change password now 2) Enable 2FA 3) Review recent logins 4) Contact security."
    elif intent == "report_fraud":
        guide = "URGENT: Advise: 1) Block card 2) File dispute 3) Note transaction details. Reassure they are protected."
    elif intent == "emergency_block":
        guide = "URGENT: Block card immediately via app or helpline. Offer replacement card."
    elif intent == "account_compromised":
        guide = "URGENT: Reset password immediately. Check registered email/phone. Contact support if locked out."
    elif any(x in intent for x in ["balance","account","bank","statement"]):
        guide = "Guide to check via app/website. Remind to keep credentials secure."
    elif any(x in intent for x in ["card","freeze","block","pin","chip"]):
        guide = "Take card issues seriously. Provide clear next steps."
    elif any(x in intent for x in ["transfer","transaction","payment","send","wire"]):
        guide = "Be precise. Confirm transaction type. Provide processing times and fees."
    elif any(x in intent for x in ["fraud","dispute","unauthorized","stolen","scam"]):
        guide = "Highest urgency. Reassure customer they are protected. Guide to report/dispute."
    elif any(x in intent for x in ["order","delivery","shipping","track","return","refund"]):
        guide = "Acknowledge concern. Provide tracking steps or resolution timeline."
    elif any(x in intent for x in ["book","flight","hotel","reservation","ticket","travel"]):
        guide = "Help with booking clearly. Confirm details. Provide next steps."
    elif any(x in intent for x in ["loan","credit","mortgage","interest","borrow"]):
        guide = "Be transparent. Use 'typically' or 'generally'. Avoid promises."
    elif any(x in intent for x in ["app","login","password","reset","otp","verify"]):
        guide = "Guide through steps clearly. Reassure issue is common."
    elif any(x in intent for x in ["greeting","hello","thank","bye","goodbye"]):
        guide = "Respond naturally and warmly. Keep brief."
    else:
        guide = "Understand the need and respond helpfully. Be clear and actionable."

    ctx = ""
    if history:
        ctx = "\nPrevious conversation:\n" + "".join(
            f"  {'Customer' if t['role']=='user' else 'Bot'}: {t['content']}\n"
            for t in history[-4:]
        ) + "\nContinue naturally:\n"

    return (
        f"You are a professional customer service chatbot for a financial and services company.\n"
        f"{ctx}Customer query: \"{query}\"\nTopic detected: {ir}\n\n"
        f"Tone: {tone}\nHow to handle: {guide}\n\n"
        f"Rules: Write 2-3 complete helpful sentences. "
        f"Do not mention intent names, confidence scores or system labels. "
        f"Sound human and natural. End with a period or exclamation mark.\n"
    )

# ─────────────────────────────────────────────
# AI RESPONSE — unchanged
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
        return f"API Error {r.status_code} — check your key."
    except Exception as e:
        return f"Connection error: {str(e)[:80]}"


def latency_stats():
    lats = st.session_state.latencies
    if not lats: return None
    return {"avg": round(sum(lats)/len(lats)), "min": min(lats), "max": max(lats)}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 14px 0; border-bottom:1px solid #e5e7eb; margin-bottom:2px;'>
        <div style='font-size:1rem; font-weight:600; color:#1a3c5e; letter-spacing:-0.2px;'>
            Query Analyzer
        </div>
        <div style='font-size:0.68rem; color:#9ca3af; margin-top:2px;'>
            Final Year Project · 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-sec'>AI Provider</div>", unsafe_allow_html=True)
    provider = st.selectbox(
        "provider_select",
        options=["groq", "gemini", "openai", "claude"],
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: x.upper(),
    )
    provider_meta = {
        "groq"  : ("Free", "console.groq.com"),
        "gemini": ("Free tier", "aistudio.google.com"),
        "openai": ("Paid", "platform.openai.com"),
        "claude": ("Paid", "console.anthropic.com"),
    }
    tier, url = provider_meta[provider]
    st.markdown(
        f"<div style='font-size:0.69rem; color:#6b7280; margin:-2px 0 8px 2px;'>"
        f"{tier} &middot; <a href='https://{url}' style='color:#1a3c5e;'>{url}</a></div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='sb-sec'>API Key</div>", unsafe_allow_html=True)

    import os as _os
    _on_cloud = _os.environ.get("STREAMLIT_SHARING_MODE") or _os.path.exists("/mount/src")
    api_key = ""
    if _on_cloud:
        try:
            sm = {"groq":"GROQ_API_KEY","gemini":"GEMINI_API_KEY","openai":"OPENAI_API_KEY","claude":"CLAUDE_API_KEY"}
            api_key = st.secrets[sm[provider]]
            st.markdown("<div style='font-size:0.69rem;color:#166534;margin-bottom:8px;'>Key loaded from secrets</div>", unsafe_allow_html=True)
        except Exception:
            api_key = ""
    if not api_key:
        api_key = st.text_input(
            "api_key_input",
            label_visibility="collapsed",
            type="password",
            placeholder="Paste your API key...",
        )
        if api_key:
            masked = api_key[:4] + "x" * min(len(api_key)-8, 10) + api_key[-4:] if len(api_key) > 8 else "x" * len(api_key)
            st.markdown(f"<div style='font-size:0.69rem;color:#166534;margin:-2px 0 6px 0;'>Key set: {masked}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#e5e7eb;margin:10px 0;'></div>", unsafe_allow_html=True)

    with st.expander("Model Paths", expanded=False):
        model_path = st.text_input(
            "BERT model folder",
            value=r"C:\Users\Sastra\Documents\project_s",
        )
        data_path = st.text_input(
            "Data folder",
            value=r"C:\Users\Sastra\Documents\project_s\clinc_oos\pre_processed",
        )

    load_btn = st.button("Load BERT Model", use_container_width=True)

    st.markdown("<div style='height:1px;background:#e5e7eb;margin:12px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sec'>Session Statistics</div>", unsafe_allow_html=True)

    total = st.session_state.total_queries
    neg   = st.session_state.sentiment_counts["negative"]
    neu   = st.session_state.sentiment_counts["neutral"]
    pos   = st.session_state.sentiment_counts["positive"]
    sec   = st.session_state.security_count
    low   = st.session_state.lowconf_count
    ls    = latency_stats()

    rows = [
        ("Total queries",     str(total), "#1a3c5e"),
        ("Negative",          str(neg),   "#b91c1c"),
        ("Neutral",           str(neu),   "#374151"),
        ("Positive",          str(pos),   "#166534"),
        ("Security alerts",   str(sec),   "#b91c1c"),
        ("Low confidence",    str(low),   "#92400e"),
    ]
    if ls:
        rows += [
            ("Avg latency", f"{ls['avg']} ms", "#1a3c5e"),
            ("Min latency", f"{ls['min']} ms", "#166534"),
            ("Max latency", f"{ls['max']} ms", "#b91c1c"),
        ]
    for label, val, color in rows:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.8rem;padding:3px 0;border-bottom:1px solid #f3f4f6;'>"
            f"<span style='color:#6b7280;'>{label}</span>"
            f"<span style='font-weight:600;color:{color};"
            f"font-family:JetBrains Mono,monospace;font-size:0.78rem;'>{val}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1px;background:#e5e7eb;margin:12px 0;'></div>", unsafe_allow_html=True)

    if st.session_state.history_log:
        df_exp   = pd.DataFrame(st.session_state.history_log)
        csv_data = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download History (CSV)",
            data=csv_data,
            file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if st.button("Clear Conversation", use_container_width=True):
        for k, v in _defaults.items():
            if isinstance(v, dict): st.session_state[k] = v.copy()
            elif isinstance(v, list): st.session_state[k] = []
            else: st.session_state[k] = v
        st.rerun()

    st.markdown("""
    <div style='margin-top:12px;padding:10px 12px;background:#f9fafb;border-radius:6px;
                border:1px solid #e5e7eb;font-size:0.69rem;color:#6b7280;line-height:1.7;'>
        <div style='font-weight:600;color:#1a3c5e;margin-bottom:4px;'>Model Info</div>
        Intent Accuracy &nbsp;: <b style='color:#166534;'>86.20%</b><br>
        Sentiment Accuracy: <b style='color:#166534;'>93.13%</b><br>
        Dataset &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : CLINC150 (151 intents)<br>
        Architecture &nbsp;&nbsp;&nbsp;: BERT Multi-task<br>
        Training time &nbsp;&nbsp;: 10.8 min, RTX 2000 Ada
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
if load_btn:
    if not api_key:
        st.warning("Enter your API key before loading.")
    else:
        with st.spinner("Loading BERT model — first load takes ~15 seconds..."):
            try:
                mdl, tok, i2i, oid, dev = load_model(model_path, data_path)
                st.session_state.update({
                    "bert_loaded": True, "model": mdl, "tokenizer": tok,
                    "id2intent": i2i, "oos_id": oid, "device": dev,
                    "provider": provider, "api_key": api_key,
                })
                st.success(f"BERT loaded — {len(i2i)} intents. Provider: {provider.upper()} ({MODELS[provider]})")
            except Exception as e:
                st.error(f"Load failed: {e}")

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="header-tags">
        <span class="htag">BERT Multi-Task</span>
        <span class="htag">151 Intents</span>
        <span class="htag">Safety Net</span>
        <span class="htag">Multi-Provider LLM</span>
    </div>
    <h1>Customer Query Analyzer</h1>
    <p>Intent classification &middot; sentiment analysis &middot; automated response generation</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_chat, col_right = st.columns([1.05, 0.95], gap="large")

# ══════════════════════════════════════════════
# CHAT COLUMN
# ══════════════════════════════════════════════
with col_chat:
    st.markdown('<div class="section-label">Chat Interface</div>', unsafe_allow_html=True)

    # Build chat HTML
    if not st.session_state.messages:
        chat_html = """
        <div class="chat-window">
            <div class="empty-state">
                <div class="text">Load the model and start a conversation</div>
                <div class="hint">
                    Try: "What is my account balance?" &nbsp;&middot;&nbsp;
                    "Someone hacked my account" &nbsp;&middot;&nbsp;
                    "I lost my card"
                </div>
            </div>
        </div>"""
    else:
        chat_html = '<div class="chat-window">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="bubble-user">{msg["content"]}</div>'
                    f'<div class="msg-meta" style="justify-content:flex-end;">'
                    f'{msg["time"]}</div>'
                )
            else:
                is_sec  = msg.get("pre_classified", False)
                is_low  = msg.get("low_confidence", False)
                bubble  = "bubble-security" if is_sec else "bubble-bot"
                i_label = msg.get("intent", "").replace("_", " ")
                s       = msg.get("sentiment", "neutral")
                s_cls   = {"negative":"t-neg","neutral":"t-neu","positive":"t-pos"}.get(s,"t-neu")
                fb      = msg.get("feedback", "")
                fb_tag  = ""
                if fb == "up":   fb_tag = ' <span class="tag t-good">Helpful</span>'
                elif fb == "down": fb_tag = ' <span class="tag t-bad">Not helpful</span>'

                tags = (
                    f'<span class="tag t-intent">{i_label}</span> '
                    f'<span class="tag {s_cls}">{SENTIMENT_LABEL.get(s, s)}</span>'
                )
                if is_sec: tags += ' <span class="tag t-sec">SECURITY ALERT</span>'
                if is_low: tags += ' <span class="tag t-low">LOW CONFIDENCE</span>'
                tags += f'{fb_tag} <span style="color:#d1d5db;font-size:0.6rem;">{msg.get("time","")} &middot; {msg.get("latency","")}</span>'

                chat_html += (
                    f'<div class="{bubble}">{msg["content"]}</div>'
                    f'<div class="msg-meta">{tags}</div>'
                )
        chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

    # Input form — text field + arrow button inline
    with st.form("chat_form", clear_on_submit=True):
        input_col, arrow_col = st.columns([11, 1])
        with input_col:
            user_input = st.text_input(
                "input_field",
                label_visibility="collapsed",
                placeholder="Type your query here...  e.g. 'What is my balance?' or 'Someone hacked my account'"
            )
        with arrow_col:
            submitted = st.form_submit_button("&#8594;", use_container_width=True)

    # Feedback for last response
    bot_msgs = [m for m in st.session_state.messages if m["role"] == "bot"]
    if bot_msgs:
        last_idx = len(st.session_state.messages) - 1
        while last_idx >= 0 and st.session_state.messages[last_idx]["role"] != "bot":
            last_idx -= 1
        if last_idx >= 0 and st.session_state.messages[last_idx].get("feedback", "") == "":
            st.markdown(
                "<div style='font-size:0.69rem;color:#9ca3af;margin:2px 0 4px 2px;"
                "font-family:JetBrains Mono,monospace;'>Was this response helpful?</div>",
                unsafe_allow_html=True
            )
            fb1, fb2, _sp = st.columns([1, 1, 6])
            with fb1:
                if st.button("Yes", key="fb_up", use_container_width=True):
                    st.session_state.messages[last_idx]["feedback"] = "up"
                    if st.session_state.history_log:
                        st.session_state.history_log[-1]["Feedback"] = "Yes"
                    st.rerun()
            with fb2:
                if st.button("No", key="fb_down", use_container_width=True):
                    st.session_state.messages[last_idx]["feedback"] = "down"
                    if st.session_state.history_log:
                        st.session_state.history_log[-1]["Feedback"] = "No"
                    st.rerun()

    # Quick examples
    st.markdown(
        "<div style='font-size:0.65rem;color:#9ca3af;margin:8px 0 5px 0;"
        "font-family:JetBrains Mono,monospace;letter-spacing:0.5px;'>QUICK EXAMPLES</div>",
        unsafe_allow_html=True
    )
    examples = [
        "What is my account balance?",
        "I lost my card, block it now",
        "Someone hacked my account",
        "Book a flight to Chennai",
        "Translate hello to French",
        "Late delivery, I am frustrated",
    ]
    eq1, eq2, eq3 = st.columns(3)
    for col, ex in zip([eq1, eq2, eq3, eq1, eq2, eq3], examples):
        with col:
            if st.button(ex, key=f"ex_{ex[:10].replace(' ','_')}", use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()

    if "_prefill" in st.session_state:
        user_input = st.session_state.pop("_prefill")
        submitted  = True

    # Process query
    if submitted and user_input and user_input.strip():
        if not st.session_state.bert_loaded:
            st.warning("Load the model first using the sidebar button.")
        else:
            with st.spinner("Analyzing..."):
                t0     = time.time()
                result = classify(
                    user_input,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    st.session_state.id2intent,
                    st.session_state.oos_id,
                    st.session_state.device,
                )
                response = get_ai_response(
                    user_input, result["intent"], result["sentiment"],
                    result["intent_confidence"],
                    st.session_state.provider, st.session_state.api_key,
                    st.session_state.conv_history,
                )
                latency = round((time.time() - t0) * 1000)
                now     = datetime.now().strftime("%H:%M")

            st.session_state.conv_history.append({"role":"user",  "content":user_input})
            st.session_state.conv_history.append({"role":"model", "content":response})
            if len(st.session_state.conv_history) > 8:
                st.session_state.conv_history = st.session_state.conv_history[-8:]

            st.session_state.messages.append({"role":"user","content":user_input,"time":now})
            st.session_state.messages.append({
                "role":"bot","content":response,
                "intent":result["intent"],"sentiment":result["sentiment"],
                "pre_classified":result["pre_classified"],
                "low_confidence":result["low_confidence"],
                "time":now,"latency":f"{latency}ms","feedback":"",
            })

            st.session_state.total_queries += 1
            st.session_state.sentiment_counts[result["sentiment"]] += 1
            st.session_state.latencies.append(latency)
            if result["pre_classified"]: st.session_state.security_count += 1
            if result["low_confidence"]: st.session_state.lowconf_count  += 1

            ik = result["intent"].replace("_", " ")
            st.session_state.intent_freq[ik] = st.session_state.intent_freq.get(ik, 0) + 1
            st.session_state.last_result = {**result, "response":response, "latency":latency, "query":user_input}

            flag = "Security" if result["pre_classified"] else ("Low conf" if result["low_confidence"] else "OK")
            st.session_state.history_log.append({
                "Time"      : now,
                "Query"     : user_input[:44]+"..." if len(user_input)>44 else user_input,
                "Intent"    : ik,
                "Confidence": f"{result['intent_confidence']*100:.1f}%",
                "Sentiment" : SENTIMENT_LABEL.get(result["sentiment"], result["sentiment"]),
                "Status"    : flag,
                "Latency"   : f"{latency}ms",
                "Feedback"  : "",
            })
            st.rerun()

# ══════════════════════════════════════════════
# ANALYTICS COLUMN
# ══════════════════════════════════════════════
with col_right:
    st.markdown('<div class="section-label">Analysis Panel</div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        r = st.session_state.last_result

        # Metric tiles
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val" style="font-size:0.76rem;line-height:1.4;word-break:break-word;">
                    {r['intent'].replace('_',' ').title()}
                </div>
                <div class="lbl">Intent</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val" style="font-size:0.9rem;">{SENTIMENT_LABEL.get(r['sentiment'], r['sentiment'])}</div>
                <div class="lbl">Sentiment</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            fl = "Security" if r["pre_classified"] else ("Low Conf" if r["low_confidence"] else "Normal")
            fv_color = "#b91c1c" if r["pre_classified"] else ("#92400e" if r["low_confidence"] else "#166534")
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val" style="font-size:0.82rem;color:{fv_color};">{fl}</div>
                <div class="lbl">Status</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # Confidence gauge
        st.markdown('<div class="section-label">Intent Confidence</div>', unsafe_allow_html=True)
        conf_pct    = round(r["intent_confidence"] * 100, 1)
        gauge_color = "#b91c1c" if r["pre_classified"] else ("#d97706" if conf_pct < 50 else "#1a3c5e")
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = conf_pct,
            number= {"suffix":"%","font":{"size":20,"color":"#111827","family":"JetBrains Mono"}},
            gauge = {
                "axis"    : {"range":[0,100],"tickwidth":1,"tickcolor":"#e5e7eb",
                             "tickfont":{"size":9,"color":"#9ca3af"}},
                "bar"     : {"color":gauge_color,"thickness":0.24},
                "bgcolor" : "#f9fafb",
                "bordercolor": "#e5e7eb",
                "borderwidth": 1,
                "steps"   : [
                    {"range":[0,40],  "color":"#fef2f2"},
                    {"range":[40,70], "color":"#fffbeb"},
                    {"range":[70,100],"color":"#f0fdf4"},
                ],
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=155,
            margin=dict(l=16,r=16,t=8,b=8),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

        # Top-3 bars
        st.markdown('<div class="section-label">Top 3 Predictions</div>', unsafe_allow_html=True)
        bar_cls = "bar-red" if r["pre_classified"] else "bar-blue"
        for name, score in r["top3_intents"]:
            st.markdown(f"""
            <div style="margin-bottom:9px;">
                <div style="display:flex;justify-content:space-between;font-size:0.77rem;margin-bottom:3px;">
                    <span style="color:#374151;">{name.replace('_',' ')}</span>
                    <span style="color:#1a3c5e;font-family:'JetBrains Mono',monospace;font-weight:600;">{score}%</span>
                </div>
                <div class="bar-track">
                    <div class="{bar_cls}" style="width:{min(score,100)}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Sentiment bar
        st.markdown('<div class="section-label">Sentiment Breakdown</div>', unsafe_allow_html=True)
        ss  = r["sentiment_scores"]
        fig = go.Figure(go.Bar(
            x=list(ss.values()), y=["Negative","Neutral","Positive"],
            orientation="h", marker_color=["#f87171","#9ca3af","#4ade80"],
            text=[f"{v}%" for v in ss.values()], textposition="auto",
            textfont=dict(color="#1f2937",size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151",family="Inter"), height=125,
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False,showticklabels=False,range=[0,115]),
            yaxis=dict(showgrid=False,tickfont=dict(size=10)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:#ffffff;
                    border:1px solid #dde1e7;border-radius:8px;">
            <div style="font-size:0.87rem;color:#6b7280;">
                Analysis results will appear after your first query.
            </div>
            <div style="font-size:0.76rem;color:#d1d5db;margin-top:5px;">
                Load model &rarr; type query &rarr; click Send
            </div>
        </div>""", unsafe_allow_html=True)

    # Session pie
    if st.session_state.total_queries > 0:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Session Sentiment</div>', unsafe_allow_html=True)
        counts = st.session_state.sentiment_counts
        fig2   = go.Figure(go.Pie(
            labels=["Negative","Neutral","Positive"],
            values=[counts["negative"],counts["neutral"],counts["positive"]],
            hole=0.55, marker_colors=["#f87171","#9ca3af","#4ade80"],
            textfont=dict(size=10),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#374151",family="Inter"),
            height=185, margin=dict(l=0,r=0,t=0,b=0), showlegend=True,
            legend=dict(orientation="h",yanchor="bottom",y=-0.22,xanchor="center",x=0.5,font=dict(size=10)),
            annotations=[dict(
                text=f"<b>{st.session_state.total_queries}</b>",
                x=0.5,y=0.5,font=dict(size=16,color="#1a3c5e"),showarrow=False
            )],
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    # Intent frequency
    if st.session_state.intent_freq:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Intent Frequency</div>', unsafe_allow_html=True)
        sorted_i = sorted(st.session_state.intent_freq.items(), key=lambda x: x[1], reverse=True)[:6]
        fig3 = go.Figure(go.Bar(
            x=[x[1] for x in sorted_i], y=[x[0] for x in sorted_i],
            orientation="h", marker_color="#1a3c5e", opacity=0.7,
            text=[x[1] for x in sorted_i], textposition="auto",
            textfont=dict(color="#ffffff",size=10),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151",family="Inter"),
            height=max(110, len(sorted_i)*30),
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False,showticklabels=False),
            yaxis=dict(showgrid=False,tickfont=dict(size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

# ─────────────────────────────────────────────
# HISTORY TABLE
# ─────────────────────────────────────────────
if st.session_state.history_log:
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Query History</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.history_log)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Query"     : st.column_config.TextColumn("Query",      width="large"),
            "Intent"    : st.column_config.TextColumn("Intent",     width="medium"),
            "Confidence": st.column_config.TextColumn("Conf",       width="small"),
            "Sentiment" : st.column_config.TextColumn("Sentiment",  width="small"),
            "Status"    : st.column_config.TextColumn("Status",     width="small"),
            "Latency"   : st.column_config.TextColumn("Latency",    width="small"),
            "Feedback"  : st.column_config.TextColumn("Feedback",   width="small"),
        }
    )