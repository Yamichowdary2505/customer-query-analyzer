"""
=============================================================
  STEP 3: BERT TESTING & EVALUATION
  AI-Based Customer Query Analyzer
=============================================================
Run: python 3_bert_testing.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "data_dir"      : r"C:\Users\Sastra\Documents\project_s\clinc_oos\pre_processed",
    "model_dir"     : r"C:\Users\Sastra\Documents\project_s\models",
    "bert_weights"  : r"C:\Users\Sastra\Documents\project_s\models\bert_best.pt",
    "max_length"    : 64,
    "batch_size"    : 128,
    "dropout"       : 0.3,
}

SENTIMENT_NAMES = ["negative", "neutral", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# MODEL DEFINITION (must match training)
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
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = self.dropout(outputs.pooler_output)
        return self.intent_classifier(cls_output), self.sentiment_classifier(cls_output)


class QueryDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts      = df["text_clean"].tolist()
        self.intents    = df["intent"].tolist()
        self.sentiments = df["sentiment"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "token_type_ids" : enc["token_type_ids"].squeeze(0),
            "intent_label"   : torch.tensor(self.intents[idx],    dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiments[idx], dtype=torch.long),
        }


if __name__ == "__main__":

    # ── Load label map ─────────────────────────
    with open(f"{CFG['data_dir']}/intent_label_map.json") as f:
        id2intent = json.load(f)
    NUM_INTENTS = len(id2intent)

    # ── Load tokenizer & model ─────────────────
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CFG["model_dir"])

    print("Loading BERT model...")
    model = MultiTaskBERT("bert-base-uncased", NUM_INTENTS, 3, CFG["dropout"])
    model.load_state_dict(torch.load(CFG["bert_weights"], map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded ✅")

    # ── Load test data ─────────────────────────
    test_df = pd.read_csv(f"{CFG['data_dir']}/test.csv")
    print(f"Test samples: {len(test_df):,}")

    test_dataset = QueryDataset(test_df, tokenizer, CFG["max_length"])
    test_loader  = DataLoader(test_dataset, batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=0)

    # ── Batch Inference ────────────────────────
    print("\nRunning inference on test set...")
    all_intent_preds, all_intent_labels       = [], []
    all_sentiment_preds, all_sentiment_labels = [], []
    all_intent_confs = []

    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids        = batch["input_ids"].to(DEVICE)
            attention_mask   = batch["attention_mask"].to(DEVICE)
            token_type_ids   = batch["token_type_ids"].to(DEVICE)

            intent_logits, sentiment_logits = model(input_ids, attention_mask, token_type_ids)

            intent_probs = torch.softmax(intent_logits, dim=-1)
            intent_confs = intent_probs.max(dim=-1).values.cpu().numpy()

            all_intent_preds.extend(intent_logits.argmax(dim=-1).cpu().numpy())
            all_intent_labels.extend(batch["intent_label"].numpy())
            all_sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().numpy())
            all_sentiment_labels.extend(batch["sentiment_label"].numpy())
            all_intent_confs.extend(intent_confs)

    inference_time = time.time() - t0
    print(f"Inference done in {inference_time:.2f}s ({inference_time/len(test_df)*1000:.1f}ms/sample)")

    # ── Metrics ────────────────────────────────
    intent_acc    = accuracy_score(all_intent_labels, all_intent_preds)
    intent_f1     = f1_score(all_intent_labels, all_intent_preds, average="weighted", zero_division=0)
    sentiment_acc = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    sentiment_f1  = f1_score(all_sentiment_labels, all_sentiment_preds, average="weighted", zero_division=0)

    print("\n" + "="*60)
    print("  TEST SET RESULTS")
    print("="*60)
    print(f"\n  Intent Classification:")
    print(f"    Accuracy  : {intent_acc:.4f}  ({intent_acc*100:.2f}%)")
    print(f"    F1-Score  : {intent_f1:.4f}  (weighted)")
    print(f"    Mean Conf : {np.mean(all_intent_confs):.4f}")

    print(f"\n  Sentiment Analysis:")
    print(f"    Accuracy  : {sentiment_acc:.4f}  ({sentiment_acc*100:.2f}%)")
    print(f"    F1-Score  : {sentiment_f1:.4f}  (weighted)")

    print(f"\n  Speed:")
    print(f"    Total     : {inference_time:.2f}s for {len(test_df)} samples")
    print(f"    Per sample: {inference_time/len(test_df)*1000:.2f}ms")

    print("\n-- Sentiment Report --")
    print(classification_report(
        all_sentiment_labels, all_sentiment_preds,
        target_names=SENTIMENT_NAMES, zero_division=0
    ))

    # ── Save results ───────────────────────────
    results_df = test_df.copy()
    results_df["pred_intent"]    = all_intent_preds
    results_df["pred_sentiment"] = all_sentiment_preds
    results_df["intent_conf"]    = [round(c, 4) for c in all_intent_confs]
    results_df["pred_intent_name"] = [id2intent[str(p)] for p in all_intent_preds]
    results_df["pred_sentiment_name"] = [SENTIMENT_NAMES[p] for p in all_sentiment_preds]
    results_df["intent_correct"] = (results_df["intent"] == results_df["pred_intent"]).astype(int)

    out_path = r"C:\Users\Sastra\Documents\project_s\bert_test_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nTest results saved → {out_path}")

    # ── Sample predictions ─────────────────────
    print("\n-- Sample Predictions --")
    for _, row in results_df.sample(8, random_state=42).iterrows():
        status = "✅" if row["intent_correct"] else "❌"
        print(f"  {status} Query     : {row['text']}")
        print(f"     True intent : {row['intent_name']}  |  Pred: {row['pred_intent_name']}  (conf: {row['intent_conf']})")
        print(f"     Sentiment   : {row['pred_sentiment_name']}")
        print()

    print("✅ BERT Testing complete! Proceed to Step 4: Gemini API setup.")
