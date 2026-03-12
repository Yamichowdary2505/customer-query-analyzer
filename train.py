"""
=============================================================
  STEP 2: BERT TRAINING — GPU OPTIMIZED (WINDOWS FIXED)
  AI-Based Customer Query Analyzer

  FIX APPLIED:
    1. num_workers set to 0 (safest fix for Windows)
    2. persistent_workers removed (only valid when num_workers > 0)
    3. All code wrapped inside if __name__ == "__main__"
    4. GradScaler updated to new non-deprecated API
=============================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "data_dir"   : r"C:\Users\Sastra\Documents\project_s\clinc_oos\pre_processed",
    "output_dir" : r"C:\Users\Sastra\Documents\project_s",

    "bert_model" : "bert-base-uncased",
    "max_length" : 64,
    "dropout"    : 0.3,

    "batch_size"  : 64,
    "epochs"      : 15,
    "lr"          : 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    "intent_loss_weight"   : 0.7,
    "sentiment_loss_weight": 0.3,

    "fp16"        : True,
    # ✅ FIX 1: num_workers = 0 on Windows
    # Windows uses "spawn" for multiprocessing (not "fork" like Linux).
    # num_workers > 0 causes child processes to re-import the script,
    # triggering the RuntimeError. Setting 0 disables multiprocessing
    # entirely — DataLoader runs in the main process. No speed loss on
    # GPU since the bottleneck is GPU compute, not data loading.
    "num_workers" : 0,
    "pin_memory"  : True,

    "save_every_epoch" : True,
    "log_every_batches": 20,
    "seed"             : 42,
}

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class QueryDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts      = df["text_clean"].tolist()
        self.intents    = df["intent"].tolist()
        self.sentiments = df["sentiment"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "token_type_ids" : enc["token_type_ids"].squeeze(0),
            "intent_label"   : torch.tensor(self.intents[idx],    dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiments[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# MODEL
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
        intent_logits    = self.intent_classifier(cls_output)
        sentiment_logits = self.sentiment_classifier(cls_output)
        return intent_logits, sentiment_logits


# ─────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler,
                device, cfg, intent_criterion, sentiment_criterion):
    model.train()
    total_loss = 0
    all_intent_preds, all_intent_labels       = [], []
    all_sentiment_preds, all_sentiment_labels = [], []
    epoch_start = time.time()

    for batch_idx, batch in enumerate(loader):
        input_ids        = batch["input_ids"].to(device)
        attention_mask   = batch["attention_mask"].to(device)
        token_type_ids   = batch["token_type_ids"].to(device)
        intent_labels    = batch["intent_label"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=cfg["fp16"]):
            intent_logits, sentiment_logits = model(
                input_ids, attention_mask, token_type_ids
            )
            intent_loss    = intent_criterion(intent_logits, intent_labels)
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            loss = (
                cfg["intent_loss_weight"]    * intent_loss +
                cfg["sentiment_loss_weight"] * sentiment_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        all_intent_preds.extend(intent_logits.argmax(dim=-1).cpu().numpy())
        all_intent_labels.extend(intent_labels.cpu().numpy())
        all_sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().numpy())
        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

        if (batch_idx + 1) % cfg["log_every_batches"] == 0:
            elapsed      = time.time() - epoch_start
            batches_done = batch_idx + 1
            batches_left = len(loader) - batches_done
            eta_secs     = (elapsed / batches_done) * batches_left
            eta_str      = str(timedelta(seconds=int(eta_secs)))
            cur_acc      = accuracy_score(all_intent_labels, all_intent_preds)
            print(
                f"    Batch {batches_done:>4}/{len(loader)} | "
                f"Loss: {total_loss/batches_done:.4f} | "
                f"Intent Acc: {cur_acc:.4f} | "
                f"ETA: {eta_str}",
                flush=True
            )

    intent_acc    = accuracy_score(all_intent_labels, all_intent_preds)
    sentiment_acc = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    epoch_time    = time.time() - epoch_start
    return total_loss / len(loader), intent_acc, sentiment_acc, epoch_time


def eval_epoch(model, loader, device, cfg, intent_criterion, sentiment_criterion):
    model.eval()
    total_loss = 0
    all_intent_preds, all_intent_labels       = [], []
    all_sentiment_preds, all_sentiment_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids        = batch["input_ids"].to(device)
            attention_mask   = batch["attention_mask"].to(device)
            token_type_ids   = batch["token_type_ids"].to(device)
            intent_labels    = batch["intent_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=cfg["fp16"]):
                intent_logits, sentiment_logits = model(
                    input_ids, attention_mask, token_type_ids
                )
                intent_loss    = intent_criterion(intent_logits, intent_labels)
                sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
                loss = (
                    cfg["intent_loss_weight"]    * intent_loss +
                    cfg["sentiment_loss_weight"] * sentiment_loss
                )

            total_loss += loss.item()
            all_intent_preds.extend(intent_logits.argmax(dim=-1).cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())
            all_sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

    intent_acc    = accuracy_score(all_intent_labels, all_intent_preds)
    sentiment_acc = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    return (
        total_loss / len(loader),
        intent_acc, sentiment_acc,
        all_intent_preds, all_sentiment_preds,
        all_intent_labels, all_sentiment_labels,
    )


# ─────────────────────────────────────────────
# ✅ FIX 2: ALL execution inside __main__ guard
# This is MANDATORY on Windows when using
# multiprocessing (even with num_workers=0,
# it's best practice to keep it here).
# ─────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(CFG["output_dir"], exist_ok=True)
    torch.manual_seed(CFG["seed"])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  BERT MULTI-TASK TRAINING — GPU VERSION")
    print("  AI-Based Customer Query Analyzer")
    print("=" * 60)
    print(f"\nDevice       : {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM         : {vram:.1f} GB")
    else:
        print("WARNING: GPU not detected!")

    print(f"Batch size   : {CFG['batch_size']}")
    print(f"Epochs       : {CFG['epochs']}")
    print(f"Learning rate: {CFG['lr']}")
    print(f"fp16         : {CFG['fp16']}")
    print(f"num_workers  : {CFG['num_workers']}  ← 0 = Windows safe mode")

    # ── Load Data ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Loading Data")
    print("=" * 60)

    train_df = pd.read_csv(f"{CFG['data_dir']}/train.csv")
    val_df   = pd.read_csv(f"{CFG['data_dir']}/validation.csv")

    with open(f"{CFG['data_dir']}/intent_label_map.json") as f:
        id2intent = json.load(f)

    NUM_INTENTS    = len(id2intent)
    NUM_SENTIMENTS = 3

    print(f"\nTrain samples : {len(train_df):,}")
    print(f"Val samples   : {len(val_df):,}")
    print(f"Intents       : {NUM_INTENTS}")
    print(f"Sentiments    : {NUM_SENTIMENTS} (negative / neutral / positive)")

    # ── Tokenizer ──────────────────────────────
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CFG["bert_model"])
    print("Tokenizer loaded ✅")

    # ── DataLoaders ────────────────────────────
    train_dataset = QueryDataset(train_df, tokenizer, CFG["max_length"])
    val_dataset   = QueryDataset(val_df,   tokenizer, CFG["max_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size  = CFG["batch_size"],
        shuffle     = True,
        num_workers = CFG["num_workers"],  # ✅ 0 = no child processes on Windows
        pin_memory  = CFG["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = CFG["batch_size"],
        shuffle     = False,
        num_workers = CFG["num_workers"],
        pin_memory  = CFG["pin_memory"],
    )

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Model ──────────────────────────────────
    print("\nLoading BERT model (bert-base-uncased)...")
    model = MultiTaskBERT(
        CFG["bert_model"], NUM_INTENTS, NUM_SENTIMENTS, CFG["dropout"]
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params : {total_params:,}")
    print(f"Trainable    : {trainable:,}")
    print("Model ready ✅")

    # ── Loss & Optimizer ───────────────────────
    intent_criterion    = nn.CrossEntropyLoss()
    sentiment_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    total_steps  = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ✅ FIX 3: Updated GradScaler API (non-deprecated form)
    scaler = torch.amp.GradScaler(enabled=CFG["fp16"])

    # ── Time Estimate ──────────────────────────
    print("\n" + "=" * 60)
    print("  Training Time Estimate (RTX Ada 2000)")
    print("=" * 60)
    print(f"  Batches per epoch : {len(train_loader)}")
    print(f"  Total epochs      : {CFG['epochs']}")
    print(f"  Est. time/epoch   : ~2-3 minutes")
    print(f"  Est. total time   : ~20-30 minutes")
    print(f"  Started at        : {datetime.now().strftime('%H:%M:%S')}")
    finish_est = datetime.now() + timedelta(minutes=25)
    print(f"  Est. finish       : {finish_est.strftime('%H:%M:%S')} (approx)")

    # ── Training Loop ──────────────────────────
    best_val_intent_acc = 0.0
    history             = []
    training_start      = time.time()

    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{CFG['epochs']}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        print("  Training...")
        train_loss, train_int_acc, train_sent_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            DEVICE, CFG, intent_criterion, sentiment_criterion
        )

        print("  Validating...")
        (val_loss, val_int_acc, val_sent_acc,
         vi_preds, vs_preds, vi_labels, vs_labels) = eval_epoch(
            model, val_loader, DEVICE, CFG,
            intent_criterion, sentiment_criterion
        )

        print(f"\n  Epoch {epoch} Results:")
        print(f"  Train -> Loss: {train_loss:.4f} | Intent Acc: {train_int_acc:.4f} | Sentiment Acc: {train_sent_acc:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f} | Intent Acc: {val_int_acc:.4f} | Sentiment Acc: {val_sent_acc:.4f}")
        print(f"  Time  -> {str(timedelta(seconds=int(epoch_time)))}")

        if torch.cuda.is_available():
            vram_used  = torch.cuda.memory_allocated(0) / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM  -> {vram_used:.1f}GB / {vram_total:.1f}GB used")

        history.append({
            "epoch"              : epoch,
            "train_loss"         : round(train_loss, 4),
            "val_loss"           : round(val_loss, 4),
            "train_intent_acc"   : round(train_int_acc, 4),
            "val_intent_acc"     : round(val_int_acc, 4),
            "train_sentiment_acc": round(train_sent_acc, 4),
            "val_sentiment_acc"  : round(val_sent_acc, 4),
            "epoch_time_mins"    : round(epoch_time / 60, 2),
        })

        if val_int_acc > best_val_intent_acc:
            best_val_intent_acc = val_int_acc
            torch.save(model.state_dict(), f"{CFG['output_dir']}/bert_best.pt")
            print(f"  ✅ Best model saved! (val intent acc: {val_int_acc:.4f})")

        if CFG["save_every_epoch"]:
            torch.save(
                model.state_dict(),
                f"{CFG['output_dir']}/bert_epoch{epoch}.pt"
            )
            print(f"  Epoch {epoch} checkpoint saved")

        elapsed_total = time.time() - training_start
        avg_per_epoch = elapsed_total / epoch
        remaining     = avg_per_epoch * (CFG["epochs"] - epoch)
        print(f"  Est. remaining: {str(timedelta(seconds=int(remaining)))}")

        pd.DataFrame(history).to_csv(
            f"{CFG['output_dir']}/training_history.csv", index=False
        )

    # ── Save Final Model ───────────────────────
    print("\n" + "=" * 60)
    print("  Saving Final Model")
    print("=" * 60)

    tokenizer.save_pretrained(CFG["output_dir"])
    with open(f"{CFG['output_dir']}/bert_config.json", "w") as f:
        json.dump(
            {**CFG, "num_intents": NUM_INTENTS, "num_sentiments": NUM_SENTIMENTS},
            f, indent=2
        )
    pd.DataFrame(history).to_csv(
        f"{CFG['output_dir']}/training_history.csv", index=False
    )

    print("Tokenizer   -> saved")
    print("Config      -> saved")
    print("History CSV -> saved")

    # ── Final Report ───────────────────────────
    total_time = time.time() - training_start

    print("\n" + "=" * 60)
    print("  FINAL VALIDATION REPORT")
    print("=" * 60)
    print(f"\nBest Val Intent Accuracy : {best_val_intent_acc:.4f}")
    print(f"Total Training Time      : {str(timedelta(seconds=int(total_time)))}")

    print("\n-- Intent Classification Report (sample 500) --")
    print(classification_report(vi_labels[:500], vi_preds[:500], zero_division=0))

    print("\n-- Sentiment Analysis Report --")
    print(classification_report(
        vs_labels, vs_preds,
        target_names=["negative", "neutral", "positive"],
        zero_division=0
    ))

    print("\n-- Training History --")
    print(pd.DataFrame(history).to_string(index=False))

    total_mins = total_time / 60
    print(f"\n{'='*60}")
    print(f"  BERT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"""
Total time : {total_mins:.1f} minutes
Saved to   : {CFG['output_dir']}

Files created:
  bert_best.pt           <- best model by val intent acc
  bert_epoch1-10.pt      <- per-epoch checkpoints
  bert_config.json       <- config used
  training_history.csv   <- loss and accuracy per epoch
  vocab.txt + tokenizer  <- tokenizer files

Next step:
  -> Run 3_t5_training.py
""")