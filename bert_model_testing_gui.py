"""
=============================================================
  BERT Query Analyzer — Interactive GUI
  AI-Based Customer Query Analyzer
=============================================================
Run: python bert_query_gui.py
"""

import os, json, threading
import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ─────────────────────────────────────────────
# CONFIG  ← adjust paths if needed
# ─────────────────────────────────────────────
CFG = {
    "data_dir"    : r"C:\Users\Sastra\Documents\project_s\clinc_oos\pre_processed",
    "model_dir"   : r"C:\Users\Sastra\Documents\project_s",
    "bert_weights": r"C:\Users\Sastra\Documents\project_s\bert_best.pt",
    "max_length"  : 64,
    "dropout"     : 0.3,
}

SENTIMENT_NAMES   = ["negative", "neutral", "positive"]
SENTIMENT_COLORS  = {"negative": "#FF6B6B", "neutral": "#FFD93D", "positive": "#6BCB77"}
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# MODEL (identical to training)
# ─────────────────────────────────────────────
class MultiTaskBERT(nn.Module):
    def __init__(self, bert_model_name, num_intents, num_sentiments, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_intents),
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_sentiments),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        out        = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls        = self.dropout(out.pooler_output)
        return self.intent_classifier(cls), self.sentiment_classifier(cls)


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────
def predict(text, model, tokenizer, id2intent):
    enc = tokenizer(
        text, max_length=CFG["max_length"],
        padding="max_length", truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        intent_logits, sentiment_logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            enc["token_type_ids"].to(DEVICE),
        )
    intent_probs    = torch.softmax(intent_logits, dim=-1).cpu().numpy()[0]
    sentiment_probs = torch.softmax(sentiment_logits, dim=-1).cpu().numpy()[0]

    top5_idx   = np.argsort(intent_probs)[::-1][:5]
    top5       = [(id2intent[str(i)], float(intent_probs[i])) for i in top5_idx]
    sentiment  = SENTIMENT_NAMES[int(np.argmax(sentiment_probs))]
    sent_conf  = float(np.max(sentiment_probs))

    return top5, sentiment, sent_conf, sentiment_probs


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
class App(tk.Tk):
    # ── palette ───────────────────────────────
    BG      = "#0F1117"
    PANEL   = "#1A1D27"
    ACCENT  = "#4F8EF7"
    ACCENT2 = "#7B5EA7"
    TEXT    = "#E8EAF0"
    MUTED   = "#6B7080"
    BORDER  = "#2A2D3A"

    def __init__(self):
        super().__init__()
        self.title("BERT Query Analyzer")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.minsize(780, 580)

        # state
        self.model     = None
        self.tokenizer = None
        self.id2intent = None
        self._history  = []

        self._build_ui()
        self._load_model_async()

    # ── UI construction ────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Header ──
        hdr = tk.Frame(self, bg=self.BG, pady=18)
        hdr.grid(row=0, column=0, sticky="ew", padx=28)
        tk.Label(hdr, text="⬡  BERT Query Analyzer",
                 bg=self.BG, fg=self.ACCENT,
                 font=("Courier New", 18, "bold")).pack(side="left")
        self.device_lbl = tk.Label(hdr, text=f"device: {DEVICE}",
                                   bg=self.BG, fg=self.MUTED,
                                   font=("Courier New", 9))
        self.device_lbl.pack(side="right", pady=4)

        # ── Body ──
        body = tk.Frame(self, bg=self.BG)
        body.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 20))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left column
        left = tk.Frame(body, bg=self.BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self._build_input_panel(left)
        self._build_results_panel(left)

        # Right column
        right = tk.Frame(body, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_history_panel(right)

    def _card(self, parent, title, row, col=0, rowspan=1, sticky="nsew"):
        frame = tk.Frame(parent, bg=self.PANEL,
                         highlightbackground=self.BORDER,
                         highlightthickness=1)
        frame.grid(row=row, column=col, rowspan=rowspan,
                   sticky=sticky, pady=(0, 10))
        tk.Label(frame, text=title, bg=self.PANEL,
                 fg=self.MUTED, font=("Courier New", 8, "bold"),
                 anchor="w", padx=14, pady=8).pack(fill="x")
        sep = tk.Frame(frame, bg=self.BORDER, height=1)
        sep.pack(fill="x")
        return frame

    def _build_input_panel(self, parent):
        card = self._card(parent, "INPUT QUERY", row=0, sticky="ew")
        parent.rowconfigure(0, weight=0)

        inner = tk.Frame(card, bg=self.PANEL, padx=14, pady=10)
        inner.pack(fill="both")
        inner.columnconfigure(0, weight=1)

        self.entry = tk.Entry(inner,
                         bg="#22253A", fg=self.TEXT, insertbackground=self.ACCENT,
                         relief="flat", font=("Courier New", 12),
                         highlightbackground=self.ACCENT, highlightthickness=1)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=10, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self._run_predict())
        self.entry.bind("<KeyRelease>", self._on_key)
        self.entry.focus_set()

        self.predict_btn = tk.Button(
            inner, text="ANALYZE →",
            bg="#3A4060", fg="#888", relief="flat",
            font=("Courier New", 10, "bold"),
            activebackground="#3A7AE0", activeforeground="white",
            cursor="hand2", padx=14, pady=10,
            state="disabled",
            command=self._run_predict,
        )
        self.predict_btn.grid(row=0, column=1)

        # char counter
        self.char_lbl = tk.Label(inner, text="0 chars",
                                 bg=self.PANEL, fg=self.MUTED,
                                 font=("Courier New", 8))
        self.char_lbl.grid(row=1, column=0, sticky="w", pady=(4, 0))

        # status bar
        self.status_var = tk.StringVar(value="⏳ Loading model…")
        tk.Label(inner, textvariable=self.status_var,
                 bg=self.PANEL, fg=self.MUTED,
                 font=("Courier New", 8)).grid(row=1, column=1, sticky="e", pady=(4, 0))

    def _build_results_panel(self, parent):
        card = self._card(parent, "RESULTS", row=1, sticky="nsew")
        parent.rowconfigure(1, weight=1)

        self.results_inner = tk.Frame(card, bg=self.PANEL, padx=14, pady=12)
        self.results_inner.pack(fill="both", expand=True)

        # Sentiment row
        sent_row = tk.Frame(self.results_inner, bg=self.PANEL)
        sent_row.pack(fill="x", pady=(0, 14))

        tk.Label(sent_row, text="SENTIMENT", bg=self.PANEL, fg=self.MUTED,
                 font=("Courier New", 8, "bold")).pack(side="left")
        self.sent_badge = tk.Label(sent_row, text="—",
                                   bg=self.PANEL, fg=self.MUTED,
                                   font=("Courier New", 11, "bold"),
                                   padx=10, pady=2)
        self.sent_badge.pack(side="left", padx=10)

        self.sent_conf_lbl = tk.Label(sent_row, text="",
                                      bg=self.PANEL, fg=self.MUTED,
                                      font=("Courier New", 8))
        self.sent_conf_lbl.pack(side="left")

        # Sentiment bars
        self.sent_bars = {}
        for name in SENTIMENT_NAMES:
            row = tk.Frame(self.results_inner, bg=self.PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name.capitalize(), bg=self.PANEL, fg=self.MUTED,
                     font=("Courier New", 8), width=9, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=self.BORDER, height=8)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 8))
            bar_fg = tk.Frame(bar_bg, bg=SENTIMENT_COLORS[name], height=8, width=0)
            bar_fg.place(x=0, y=0, relheight=1)
            pct_lbl = tk.Label(row, text="0%", bg=self.PANEL, fg=self.MUTED,
                               font=("Courier New", 8), width=5)
            pct_lbl.pack(side="right")
            self.sent_bars[name] = (bar_bg, bar_fg, pct_lbl)

        # Divider
        tk.Frame(self.results_inner, bg=self.BORDER, height=1).pack(fill="x", pady=12)

        # Top-5 intents
        tk.Label(self.results_inner, text="TOP-5 INTENTS",
                 bg=self.PANEL, fg=self.MUTED,
                 font=("Courier New", 8, "bold")).pack(anchor="w", pady=(0, 8))

        self.intent_rows = []
        for i in range(5):
            row = tk.Frame(self.results_inner, bg=self.PANEL)
            row.pack(fill="x", pady=3)

            rank_lbl = tk.Label(row, text=f"#{i+1}", bg=self.PANEL, fg=self.MUTED,
                                font=("Courier New", 8), width=3, anchor="w")
            rank_lbl.pack(side="left")

            name_lbl = tk.Label(row, text="—", bg=self.PANEL, fg=self.TEXT,
                                font=("Courier New", 9),
                                anchor="w")
            name_lbl.pack(side="left", fill="x", expand=True, padx=(4, 8))

            bar_bg = tk.Frame(row, bg=self.BORDER, height=6, width=120)
            bar_bg.pack(side="right", padx=(0, 6))
            bar_bg.pack_propagate(False)
            bar_fg = tk.Frame(bar_bg, bg=self.ACCENT, height=6, width=0)
            bar_fg.place(x=0, y=0, relheight=1)

            conf_lbl = tk.Label(row, text="", bg=self.PANEL, fg=self.ACCENT,
                                font=("Courier New", 8, "bold"), width=6)
            conf_lbl.pack(side="right")

            self.intent_rows.append((name_lbl, conf_lbl, bar_bg, bar_fg))

    def _build_history_panel(self, parent):
        card = self._card(parent, "HISTORY", row=0, sticky="nsew")
        parent.rowconfigure(0, weight=1)

        top = tk.Frame(card, bg=self.PANEL, padx=10, pady=6)
        top.pack(fill="x")
        tk.Button(top, text="Clear", bg=self.PANEL, fg=self.MUTED,
                  relief="flat", font=("Courier New", 8),
                  cursor="hand2", command=self._clear_history).pack(side="right")

        self.hist_frame = tk.Frame(card, bg=self.PANEL)
        self.hist_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.hist_canvas = tk.Canvas(self.hist_frame, bg=self.PANEL,
                                     highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.hist_frame, orient="vertical",
                                  command=self.hist_canvas.yview)
        self.hist_inner = tk.Frame(self.hist_canvas, bg=self.PANEL)
        self.hist_inner.bind("<Configure>",
            lambda e: self.hist_canvas.configure(
                scrollregion=self.hist_canvas.bbox("all")))

        self.hist_canvas.create_window((0, 0), window=self.hist_inner, anchor="nw")
        self.hist_canvas.configure(yscrollcommand=scrollbar.set)
        self.hist_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ── Model loading ──────────────────────────
    def _load_model_async(self):
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

    def _load_model(self):
        try:
            label_path = f"{CFG['data_dir']}/intent_label_map.json"
            print(f"[DEBUG] Loading label map from: {label_path}")
            with open(label_path) as f:
                self.id2intent = json.load(f)
            print(f"[DEBUG] {len(self.id2intent)} intents loaded")

            print(f"[DEBUG] Loading tokenizer from: {CFG['model_dir']}")
            self.tokenizer = BertTokenizer.from_pretrained(CFG["model_dir"])

            print(f"[DEBUG] Building model…")
            self.model = MultiTaskBERT("bert-base-uncased",
                                       len(self.id2intent), 3, CFG["dropout"])

            print(f"[DEBUG] Loading weights from: {CFG['bert_weights']}")
            state = torch.load(CFG["bert_weights"], map_location=DEVICE,
                               weights_only=False)
            self.model.load_state_dict(state)
            self.model = self.model.to(DEVICE)
            self.model.eval()
            print("[DEBUG] Model ready ✅")
            self.after(0, self._on_model_ready)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] Model load failed:\n{tb}")
            self.after(0, lambda: self.status_var.set(f"❌ Load error: {e}"))

    def _on_model_ready(self):
        self.status_var.set(f"✅ Model ready — {len(self.id2intent)} intents  |  type a query and press Enter")
        self.predict_btn.config(state="normal", bg=self.ACCENT, fg="white")

    # ── Prediction ─────────────────────────────
    def _on_key(self, _=None):
        txt = self.entry.get()
        self.char_lbl.config(text=f"{len(txt)} chars")

    def _run_predict(self):
        if self.model is None:
            self.status_var.set("⏳ Model still loading, please wait…")
            return
        text = self.entry.get().strip()
        print(f"[DEBUG] Query text: '{text}'")
        if not text:
            self.status_var.set("⚠️ Please type a query first")
            return
        self.status_var.set("⚡ Analyzing…")
        self.predict_btn.config(state="disabled", bg="#3A4060", fg="#888")
        t = threading.Thread(target=self._predict_thread, args=(text,), daemon=True)
        t.start()

    def _predict_thread(self, text):
        import time, traceback
        try:
            t0 = time.time()
            top5, sentiment, sent_conf, sent_probs = predict(
                text, self.model, self.tokenizer, self.id2intent)
            elapsed = (time.time() - t0) * 1000
            print(f"[DEBUG] OK — intent={top5[0][0]}, sentiment={sentiment}, {elapsed:.0f}ms")
            self.after(0, lambda: self._update_ui(
                text, top5, sentiment, sent_conf, sent_probs, elapsed))
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Prediction failed:\n{tb}")
            err_msg = str(e)
            self.after(0, lambda: self.status_var.set(f"❌ Error: {err_msg}"))
            self.after(0, lambda: self.predict_btn.config(state="normal"))

    def _update_ui(self, text, top5, sentiment, sent_conf, sent_probs, elapsed_ms):
        # Sentiment badge
        color = SENTIMENT_COLORS[sentiment]
        self.sent_badge.config(text=sentiment.upper(), fg=color,
                               bg=self._hex_dim(color, 0.12))
        self.sent_conf_lbl.config(text=f"{sent_conf*100:.1f}%")

        # Sentiment bars
        self.update_idletasks()
        for i, name in enumerate(SENTIMENT_NAMES):
            bg, fg, lbl = self.sent_bars[name]
            prob = float(sent_probs[i])
            bg.update_idletasks()
            w = int(bg.winfo_width() * prob)
            fg.place(x=0, y=0, relheight=1, width=max(w, 0))
            lbl.config(text=f"{prob*100:.0f}%")

        # Intent rows
        best_conf = top5[0][1] if top5 else 1
        for i, (name_lbl, conf_lbl, bar_bg, bar_fg) in enumerate(self.intent_rows):
            if i < len(top5):
                name, conf = top5[i]
                name_lbl.config(text=name.replace("_", " "), fg=self.TEXT if i == 0 else self.MUTED)
                conf_lbl.config(text=f"{conf*100:.1f}%",
                                fg=self.ACCENT if i == 0 else self.MUTED)
                bar_bg.update_idletasks()
                w = int(bar_bg.winfo_width() * (conf / max(best_conf, 0.001)))
                bar_fg.config(bg=self.ACCENT if i == 0 else "#3A4060")
                bar_fg.place(x=0, y=0, relheight=1, width=max(w, 0))
            else:
                name_lbl.config(text="—")
                conf_lbl.config(text="")
                bar_fg.place(x=0, y=0, relheight=1, width=0)

        self.status_var.set(f"✅ Done in {elapsed_ms:.0f}ms")
        self.predict_btn.config(state="normal", bg=self.ACCENT, fg="white")

        # Add to history
        self._add_history(text, top5[0][0] if top5 else "?",
                          top5[0][1] if top5 else 0, sentiment)

    def _add_history(self, text, intent, conf, sentiment):
        self._history.insert(0, (text, intent, conf, sentiment))
        # rebuild history list
        for w in self.hist_inner.winfo_children():
            w.destroy()
        for text, intent, conf, sentiment in self._history[:40]:
            row = tk.Frame(self.hist_inner, bg=self.BORDER, pady=8, padx=10)
            row.pack(fill="x", pady=(0, 3))
            col = SENTIMENT_COLORS[sentiment]
            tk.Label(row, text=f"● {sentiment[:3].upper()}",
                     bg=self.BORDER, fg=col,
                     font=("Courier New", 7, "bold")).pack(anchor="w")
            tk.Label(row, text=text[:54] + ("…" if len(text) > 54 else ""),
                     bg=self.BORDER, fg=self.TEXT,
                     font=("Courier New", 9), wraplength=220, anchor="w", justify="left"
                     ).pack(anchor="w")
            tk.Label(row, text=f"{intent.replace('_',' ')}  {conf*100:.0f}%",
                     bg=self.BORDER, fg=self.ACCENT,
                     font=("Courier New", 8)).pack(anchor="w")
            # click to re-load query
            def _reload(t=text):
                self.query_var.set(t)
                self._on_key()
            row.bind("<Button-1>", lambda e, fn=_reload: fn())
            for child in row.winfo_children():
                child.bind("<Button-1>", lambda e, fn=_reload: fn())

    def _clear_history(self):
        self._history.clear()
        for w in self.hist_inner.winfo_children():
            w.destroy()

    # ── Utility ────────────────────────────────
    @staticmethod
    def _hex_dim(hex_color, alpha):
        """Blend hex color with dark background at given alpha."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        br, bg_, bb = 26, 29, 39   # PANEL bg
        r2 = int(br + (r - br) * alpha)
        g2 = int(bg_ + (g - bg_) * alpha)
        b2 = int(bb + (b - bb) * alpha)
        return f"#{r2:02x}{g2:02x}{b2:02x}"


# ─────────────────────────────────────────────
if __name__ == "__main__":
    style = ttk.Style()
    app = App()
    app.geometry("980x640")
    app.mainloop()