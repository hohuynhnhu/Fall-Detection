"""
train_sitting.py
Train PoseTransformer binary classifier: Sitting vs Non_Sitting
K-Fold cross validation + save best model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = r"D:\luan_van\Fall-Detection\train_sitting\dataset"
OUTPUT_DIR = r"D:\luan_van\Fall-Detection\train_sitting"

NUM_FRAMES  = 64
INPUT_DIM   = 200
NUM_CLASSES = 2
N_FOLDS     = 5
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-4
WEIGHT_DECAY= 1e-4
PATIENCE    = 10          # early stopping
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────

class PoseTransformer(nn.Module):
    """
    Giống kiến trúc TransformerEngine hiện tại (fall detection).
    Input : (batch, num_frames, input_dim)
    Output: (batch, num_classes)
    """
    def __init__(
        self,
        input_dim: int   = INPUT_DIM,
        num_frames: int  = NUM_FRAMES,
        d_model: int     = 128,
        nhead: int       = 4,
        num_layers: int  = 2,
        dim_ff: int      = 256,
        dropout: float   = 0.1,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, num_frames, d_model) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_ff,
            dropout        = dropout,
            batch_first    = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x) + self.pos_enc
        x = self.transformer(x)         # (B, T, d_model)
        x = self.norm(x.mean(dim=1))    # global avg pool → (B, d_model)
        x = self.dropout(x)
        return self.classifier(x)       # (B, num_classes)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int64)
    print(f"Loaded: X={X.shape}  y={y.shape}")
    print(f"  Sitting={int((y==1).sum())}  Non_Sitting={int((y==0).sum())}")
    return X, y


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss   = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        preds       = logits.argmax(1)
        correct    += (preds == yb).sum().item()
        total      += len(yb)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ── K-Fold Train ──────────────────────────────────────────────────────────────

def run_kfold(X, y):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset  = TensorDataset(X_tensor, y_tensor)

    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs   = []
    best_acc    = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "sitting_model_best.pt")

    print(f"\n{'='*55}")
    print(f"  Device : {DEVICE}")
    print(f"  Folds  : {N_FOLDS}")
    print(f"  Epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"{'='*55}\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"{'─'*55}")
        print(f"  FOLD {fold}/{N_FOLDS}  "
              f"(train={len(train_idx)} / val={len(val_idx)})")
        print(f"{'─'*55}")

        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(val_idx),
        )

        model     = PoseTransformer().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )

        best_fold_acc  = 0.0
        no_improve     = 0
        best_fold_state= None

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
            vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion)
            scheduler.step()

            marker = ""
            if vl_acc > best_fold_acc:
                best_fold_acc   = vl_acc
                best_fold_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
                no_improve = 0
                marker = " ★"
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1 or marker:
                print(f"    Ep {epoch:3d} | "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                      f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f} | "
                      f"{time.time()-t0:.1f}s{marker}")

            if no_improve >= PATIENCE:
                print(f"    Early stop tại epoch {epoch}")
                break

        # Evaluate best state trên val
        model.load_state_dict(best_fold_state)
        _, fold_acc, preds, labels = eval_epoch(model, val_loader, criterion)
        fold_accs.append(fold_acc)

        print(f"\n  Fold {fold} best val_acc : {fold_acc:.4f}")
        print(classification_report(
            labels, preds,
            target_names=["Non_Sitting", "Sitting"],
            digits=4,
        ))
        cm = confusion_matrix(labels, preds)
        print(f"  Confusion matrix:\n{cm}\n")

        # Lưu model tốt nhất overall
        if fold_acc > best_acc:
            best_acc = fold_acc
            torch.save({
                "model_state_dict": best_fold_state,
                "fold":             fold,
                "val_acc":          fold_acc,
                "config": {
                    "num_frames": NUM_FRAMES,
                    "input_dim":  INPUT_DIM,
                    "num_classes": NUM_CLASSES,
                    "d_model":    128,
                    "nhead":      4,
                    "num_layers": 2,
                    "dim_ff":     256,
                    "dropout":    0.1,
                },
            }, best_model_path)
            print(f"  ✓ Saved best model (fold {fold}, acc={fold_acc:.4f})")

    # ── Summary ──────────────────────────────────────────────────────────────
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)

    print(f"\n{'='*55}")
    print(f"  K-FOLD RESULTS ({N_FOLDS} folds)")
    for i, a in enumerate(fold_accs, 1):
        print(f"    Fold {i}: {a:.4f}")
    print(f"  ─────────────────────────")
    print(f"  Mean : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Best : {max(fold_accs):.4f}  (model saved)")
    print(f"{'='*55}")

    # Lưu kết quả
    results = {
        "fold_accuracies": [float(a) for a in fold_accs],
        "mean_accuracy":   float(mean_acc),
        "std_accuracy":    float(std_acc),
        "best_accuracy":   float(max(fold_accs)),
        "model_path":      best_model_path,
        "config": {
            "num_frames": NUM_FRAMES,
            "input_dim":  INPUT_DIM,
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "n_folds":    N_FOLDS,
        },
    }
    result_path = os.path.join(OUTPUT_DIR, "kfold_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {result_path}")
    print(f"  Best model  : {best_model_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_data()
    run_kfold(X, y)