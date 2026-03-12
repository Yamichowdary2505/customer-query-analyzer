"""
=============================================================
  STEP 1: PREPROCESSING
  AI-Based Customer Query Analyzer
=============================================================

PIPELINE:
  Customer Query
        ↓
     BERT  →  intent + sentiment   (trained on this data)
        ↓
   Gemini API  →  response         (no training needed)

WHAT THIS SCRIPT DOES:
  ✅ Loads real CLINC150 dataset (train/val/test)
  ✅ Cleans text
  ✅ Labels sentiment using pre-trained RoBERTa
  ✅ Saves clean CSVs for BERT training

WHAT THIS SCRIPT DOES NOT DO:
  ❌ No T5 targets
  ❌ No synthetic/hand-written responses
  ❌ No templates

Run: python 1_preprocessing_v2.py
"""

import os
import re
import json
import torch
import pandas as pd
from transformers import pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\Sastra\Documents\project_s\clinc_oos"
OUTPUT_DIR   = r"C:\Users\Sastra\Documents\project_s\clinc_oos\pre_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_LABEL_MAP = {
    "positive": "positive",
    "neutral" : "neutral",
    "negative": "negative",
    "label_0" : "negative",
    "label_1" : "neutral",
    "label_2" : "positive",
}
SENTIMENT_ID_MAP = {"negative": 0, "neutral": 1, "positive": 2}

DEVICE     = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 64

# ─────────────────────────────────────────────
# 1. LOAD CLINC150 DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Loading CLINC150 Dataset")
print("=" * 60)

def load_clinc():
    """
    Tries to load from your local folder first.
    Falls back to HuggingFace download if local load fails.
    """
    try:
        from datasets import load_from_disk
        print(f"\nLoading from local path: {DATASET_ROOT}")
        ds = load_from_disk(DATASET_ROOT)
        assert "train" in ds and "validation" in ds and "test" in ds
        print("Local dataset loaded ✅")
        return ds
    except Exception as e:
        print(f"Local load failed: {e}")
        print("Downloading CLINC150 from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("clinc/clinc_oos", "plus")
        print("Download complete ✅")
        return ds

dataset = load_clinc()
print(f"\nDataset structure:\n{dataset}")

# Convert splits to DataFrames
train_df = pd.DataFrame(dataset["train"])
val_df   = pd.DataFrame(dataset["validation"])
test_df  = pd.DataFrame(dataset["test"])

print(f"\nSplit sizes:")
print(f"  Train      : {len(train_df):,} samples")
print(f"  Validation : {len(val_df):,} samples")
print(f"  Test       : {len(test_df):,} samples")
print(f"  Total      : {len(train_df) + len(val_df) + len(test_df):,} samples")

# ─────────────────────────────────────────────
# 2. BUILD INTENT LABEL MAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Building Intent Label Map")
print("=" * 60)

intent_feature = dataset["train"].features["intent"]
id2intent = {str(i): name for i, name in enumerate(intent_feature.names)}
intent2id = {v: int(k) for k, v in id2intent.items()}

# Save intent map — needed by BERT training and Gemini response scripts
with open(f"{OUTPUT_DIR}/intent_label_map.json", "w") as f:
    json.dump(id2intent, f, indent=2)

print(f"\nTotal intents : {len(id2intent)}")
print(f"Sample intents: {list(id2intent.values())[:8]}...")
print(f"Saved → {OUTPUT_DIR}/intent_label_map.json ✅")

# Add intent name column to all splits
for df in [train_df, val_df, test_df]:
    df["intent_name"] = df["intent"].map(lambda x: id2intent[str(x)])

# ─────────────────────────────────────────────
# 3. TEXT CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Cleaning Text")
print("=" * 60)

def clean_text(text: str) -> str:
    """
    Normalizes raw customer query text:
      - Lowercase
      - Collapse multiple spaces
      - Remove unusual characters (keep apostrophes, hyphens, punctuation)
      - Reduce repeated characters (sooooo → soo)
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\'\-\?\!\.,]", "", text)
    text = re.sub(r"(\w)\1{3,}", r"\1\1", text)
    return text

for df in [train_df, val_df, test_df]:
    df["text_clean"] = df["text"].apply(clean_text)

# Show a few examples
print("\nCleaning examples:")
for _, row in train_df.sample(3, random_state=42).iterrows():
    print(f"  Original : {row['text']}")
    print(f"  Cleaned  : {row['text_clean']}")
    print()

print("Text cleaning complete ✅")

# ─────────────────────────────────────────────
# 4. SENTIMENT LABELING — RoBERTa
#
#  Uses cardiffnlp/twitter-roberta-base-sentiment-latest
#  - Real pre-trained model, trained on 58M tweets
#  - 3 classes: negative / neutral / positive
#  - No rules, no keywords — fully learned
#
#  This gives BERT real sentiment labels to train on.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Sentiment Labeling (RoBERTa)")
print("=" * 60)

print(f"\nLoading sentiment model: {SENTIMENT_MODEL}")
print("(Downloads ~500MB on first run, cached after that)")

sentiment_pipe = pipeline(
    task       = "text-classification",
    model      = SENTIMENT_MODEL,
    tokenizer  = SENTIMENT_MODEL,
    device     = DEVICE,
    batch_size = BATCH_SIZE,
    truncation = True,
    max_length = 128,
    top_k      = 1,
)
print("Sentiment model loaded ✅")


def run_sentiment(texts: list) -> tuple:
    """
    Runs sentiment inference on a list of texts.
    Handles both output formats the pipeline may return:
      - list-of-list: [[{"label": ..., "score": ...}], ...]
      - list-of-dict: [{"label": ..., "score": ...}, ...]

    Returns three parallel lists:
      names  → ["neutral", "negative", ...]
      ids    → [1, 0, ...]
      confs  → [0.97, 0.88, ...]
    """
    results = sentiment_pipe(texts)
    names, ids, confs = [], [], []

    for r in results:
        item      = r[0] if isinstance(r, list) else r
        raw_label = item["label"].lower()
        name      = SENTIMENT_LABEL_MAP.get(raw_label, "neutral")
        names.append(name)
        ids.append(SENTIMENT_ID_MAP[name])
        confs.append(round(item["score"], 4))

    return names, ids, confs


print("\nRunning sentiment on all splits...")

for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"  Processing {split_name} ({len(df):,} samples)...", flush=True)
    names, ids, confs = run_sentiment(df["text_clean"].tolist())
    df["sentiment_name"]       = names
    df["sentiment"]            = ids
    df["sentiment_confidence"] = confs
    print(f"  {split_name} done ✅")

print("\nSentiment labeling complete ✅")

# Distribution report
print("\n── Sentiment Distribution (Train) ──")
dist = train_df["sentiment_name"].value_counts()
for label, count in dist.items():
    pct = count / len(train_df) * 100
    print(f"  {label:10s} : {count:,} samples ({pct:.1f}%)")

print(f"\n  Mean confidence  : {train_df['sentiment_confidence'].mean():.4f}")
print(f"  Low conf (<0.60) : {(train_df['sentiment_confidence'] < 0.6).sum()} samples")
print(f"  High conf (>0.90): {(train_df['sentiment_confidence'] > 0.9).sum()} samples")

# ─────────────────────────────────────────────
# 5. SAVE PROCESSED DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Saving Processed Data")
print("=" * 60)

# Only real columns — no synthetic T5 targets
COLUMNS = [
    "text",           # original raw query
    "text_clean",     # cleaned query (fed to BERT)
    "intent",         # intent ID number (BERT label)
    "intent_name",    # intent readable name
    "sentiment",      # sentiment ID number (BERT label)
    "sentiment_name", # sentiment readable name
    "sentiment_confidence",  # how confident RoBERTa was
]

train_df[COLUMNS].to_csv(f"{OUTPUT_DIR}/train.csv",      index=False)
val_df[COLUMNS].to_csv(  f"{OUTPUT_DIR}/validation.csv", index=False)
test_df[COLUMNS].to_csv( f"{OUTPUT_DIR}/test.csv",       index=False)

print(f"\n  train.csv      → {len(train_df):,} rows")
print(f"  validation.csv → {len(val_df):,} rows")
print(f"  test.csv       → {len(test_df):,} rows")
print(f"\n  Saved to: {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# 6. FINAL VERIFICATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: Verification")
print("=" * 60)

# Reload and verify
verify_df = pd.read_csv(f"{OUTPUT_DIR}/train.csv")
print(f"\nReloaded train.csv — shape: {verify_df.shape}")
print(f"Columns: {list(verify_df.columns)}")
print(f"\nSample rows:")
print(verify_df[["text_clean", "intent_name", "sentiment_name", "sentiment_confidence"]].sample(5, random_state=42).to_string(index=False))

# Check for any nulls
nulls = verify_df.isnull().sum().sum()
print(f"\nNull values in train.csv: {nulls} {'✅' if nulls == 0 else '⚠️ Check these!'}")

print("\n" + "=" * 60)
print("  ✅ PREPROCESSING COMPLETE")
print("=" * 60)
print(f"""
What was produced:
  processed_data/
    ├── train.csv            ({len(train_df):,} rows)
    ├── validation.csv       ({len(val_df):,} rows)
    ├── test.csv             ({len(test_df):,} rows)
    └── intent_label_map.json ({len(id2intent)} intents)

Next step:
  → Run 2_bert_training.py to train intent + sentiment classifier
""")
