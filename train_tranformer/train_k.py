"""
train_transformer_kfold.py
Train Transformer với K-Fold Cross Validation
Output: 4 sơ đồ đánh giá mô hình cho báo cáo
  1. Confusion Matrix
  2. ROC Curve + AUC
  3. Bar chart Accuracy từng fold
  4. Bar chart Precision/Recall/F1/Specificity
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os
import json
import copy

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR     = r"D:\luan_van\Fall-Detection\train_tranformer\dataset"
SAVE_DIR     = r"D:\luan_van\Fall-Detection\train_tranformer\dataset\checkpoints1"
FIGURE_DIR   = os.path.join(SAVE_DIR, "figures")

D_MODEL      = 128
NHEAD        = 4
NUM_LAYERS   = 2
DROPOUT      = 0.4
BATCH_SIZE   = 16
EPOCHS       = 100
LR           = 5e-4
WEIGHT_DECAY = 5e-3
PATIENCE     = 15
LABEL_SMOOTH = 0.1
N_FOLDS      = 5

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

COLORS = {
    "primary":   "#2563EB",
    "secondary": "#10B981",
    "danger":    "#EF4444",
    "warning":   "#F59E0B",
    "gray":      "#6B7280",
    "light":     "#F3F4F6",
    "fall":      "#EF4444",
    "non_fall":  "#10B981",
}

# ── Load config ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config loaded:")
        print(f"  num_frames = {cfg['num_frames']}")
        print(f"  input_dim  = {cfg['input_dim']}")
        print(f"  fusion     = {cfg.get('fusion', False)}")
        return cfg
    else:
        print("WARNING: config.json không tìm thấy, dùng default")
        return {"num_frames": 64, "input_dim": 200, "fusion": True}

# ── Dataset ───────────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(X: np.ndarray, y: np.ndarray,
            input_dim: int, n_aug: int = 3) -> tuple:
    X_aug, y_aug = [X], [y]
    for _ in range(n_aug):
        X_new = X.copy()
        mask     = np.random.rand(len(X_new)) > 0.5
        mask_idx = np.where(mask)[0]
        if len(mask_idx) > 0:
            X_flip    = X_new.copy()
            x_indices = np.arange(0, input_dim, 4)
            idx = np.ix_(mask_idx, range(X_new.shape[1]), x_indices.tolist())
            X_flip[idx] = 1.0 - X_flip[idx]
            X_new = X_flip
        factor    = np.random.choice([0.8, 0.9, 1.1, 1.2])
        T         = X_new.shape[1]
        src_len   = max(1, int(T * factor))
        src_idx   = np.linspace(0, T - 1, src_len)
        src_idx   = np.clip(src_idx, 0, T - 1).astype(int)
        final_idx = np.linspace(0, len(src_idx) - 1, T, dtype=int)
        X_new     = X_new[:, src_idx, :][:, final_idx, :]
        X_new     = X_new + np.random.normal(0, 0.01, X_new.shape)
        drop_mask = np.random.rand(X_new.shape[0], X_new.shape[1]) < 0.1
        for bi in range(X_new.shape[0]):
            X_new[bi, drop_mask[bi], :] = 0.0
        X_aug.append(X_new.astype(np.float32))
        y_aug.append(y.copy())
    return np.concatenate(X_aug), np.concatenate(y_aug)

# ── Model ─────────────────────────────────────────────────────────────────────

class PoseTransformer(nn.Module):
    def __init__(self, input_dim: int, num_frames: int,
                 d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dropout=DROPOUT,
                 num_classes=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_emb   = nn.Parameter(torch.randn(1, num_frames + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        B   = x.size(0)
        x   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_emb
        x   = self.transformer(x)
        return self.head(x[:, 0])

# ── Train 1 fold ──────────────────────────────────────────────────────────────

def train_one_fold(X_train, y_train, X_val, y_val,
                   input_dim, num_frames, device, fold_idx):
    X_tr, y_tr = augment(X_train, y_train, input_dim=input_dim, n_aug=3)
    print(f"  Train sau augment: {len(X_tr)} samples "
          f"(fall={int((y_tr==1).sum())} non_fall={int((y_tr==0).sum())})")

    train_dl = DataLoader(PoseDataset(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(PoseDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = PoseTransformer(input_dim=input_dim, num_frames=num_frames).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    print(f"\n  {'Ep':>4} {'Loss':>7} {'Tr%':>7} {'Val%':>7}")
    print(f"  {'-'*30}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == yb).sum().item()

        scheduler.step()
        train_acc  = correct / len(train_dl.dataset) * 100
        train_loss = total_loss / len(train_dl)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_correct += model(xb).argmax(1).eq(yb).sum().item()

        val_acc = val_correct / len(val_dl.dataset) * 100
        marker  = " ←" if val_acc > best_val_acc else ""
        if epoch % 5 == 0 or val_acc > best_val_acc:
            print(f"  {epoch:>4} {train_loss:>7.4f} "
                  f"{train_acc:>6.1f}% {val_acc:>6.1f}%{marker}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping tại epoch {epoch}")
                break

    return best_val_acc, best_state, val_dl

# ── Plot functions ────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Sơ đồ 1: Confusion Matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["non_fall", "fall"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm_norm, annot=False, fmt=".1f",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax,
        linewidths=2, linecolor="white",
        cbar_kws={"label": "Tỷ lệ (%)"},
    )

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j]
            color = "white" if pct > 60 else "#1e293b"
            ax.text(j + 0.5, i + 0.45, f"{count}",
                    ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
            ax.text(j + 0.5, i + 0.62, f"({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, color=color)

    # Label góc
    corner_labels = [
        (0.5, 0.15, "TN", COLORS["non_fall"]),
        (1.5, 0.15, "FP", COLORS["danger"]),
        (0.5, 1.15, "FN", COLORS["warning"]),
        (1.5, 1.15, "TP", COLORS["non_fall"]),
    ]
    for x, y, lbl, c in corner_labels:
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=9, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.15))

    ax.set_title("Ma trận nhầm lẫn (Confusion Matrix)\nTổng hợp 5 folds", pad=15)
    ax.set_xlabel("Nhãn dự đoán", labelpad=10)
    ax.set_ylabel("Nhãn thực tế", labelpad=10)
    ax.tick_params(labelsize=11)

    tn, fp, fn, tp = cm.ravel()
    stats = (f"Sensitivity: {tp/(tp+fn)*100:.1f}%  |  "
             f"Specificity: {tn/(tn+fp)*100:.1f}%  |  "
             f"Miss Rate: {fn/(fn+tp)*100:.1f}%")
    fig.text(0.5, -0.04, stats, ha="center", fontsize=9,
             color=COLORS["gray"],
             bbox=dict(boxstyle="round", facecolor=COLORS["light"], alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_roc_curve(all_labels: list, all_probs: list, save_path: str):
    """Sơ đồ 2: ROC Curve"""
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])

    # ROC curve
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], color=COLORS["gray"],
            lw=1.5, linestyle="--", label="Random Classifier")

    # Highlight optimal point (closest to top-left)
    dist      = np.sqrt(fpr**2 + (1 - tpr)**2)
    opt_idx   = np.argmin(dist)
    ax.scatter(fpr[opt_idx], tpr[opt_idx],
               s=100, color=COLORS["danger"], zorder=5,
               label=f"Optimal point ({fpr[opt_idx]:.3f}, {tpr[opt_idx]:.3f})")
    ax.annotate(f"  Optimal\n  FPR={fpr[opt_idx]:.3f}\n  TPR={tpr[opt_idx]:.3f}",
                xy=(fpr[opt_idx], tpr[opt_idx]),
                xytext=(fpr[opt_idx] + 0.12, tpr[opt_idx] - 0.12),
                fontsize=8, color=COLORS["danger"],
                arrowprops=dict(arrowstyle="->", color=COLORS["danger"]))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", labelpad=10)
    ax.set_ylabel("True Positive Rate (TPR)", labelpad=10)
    ax.set_title("ROC Curve — Fall Detection\nTổng hợp 5 folds", pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # AUC badge
    ax.text(0.97, 0.08, f"AUC = {roc_auc:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, fontweight="bold", color=COLORS["primary"],
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=COLORS["primary"], alpha=0.1,
                      edgecolor=COLORS["primary"]))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_fold_accuracy(fold_accs: list, save_path: str):
    """Sơ đồ 3: Bar chart Accuracy từng fold"""
    fig, ax = plt.subplots(figsize=(7, 5))

    folds      = [f"Fold {i+1}" for i in range(len(fold_accs))]
    mean_acc   = np.mean(fold_accs)
    colors_bar = [COLORS["primary"] if a < mean_acc else COLORS["secondary"]
                  for a in fold_accs]

    bars = ax.bar(folds, fold_accs, color=colors_bar,
                  width=0.55, edgecolor="white", linewidth=1.5,
                  zorder=3)

    # Value labels trên cột
    for bar, acc in zip(bars, fold_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Mean line
    ax.axhline(mean_acc, color=COLORS["danger"], lw=2,
               linestyle="--", zorder=4,
               label=f"Mean = {mean_acc:.1f}% ± {np.std(fold_accs):.1f}%")

    ax.fill_between(
        [-0.5, len(fold_accs) - 0.5],
        mean_acc - np.std(fold_accs),
        mean_acc + np.std(fold_accs),
        alpha=0.12, color=COLORS["danger"], zorder=2,
    )

    ax.set_ylim([max(0, min(fold_accs) - 5), 101])
    ax.set_ylabel("Validation Accuracy (%)", labelpad=10)
    ax.set_title(f"Accuracy từng Fold — {N_FOLDS}-Fold Cross Validation", pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.set_axisbelow(True)

    # Legend màu
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["secondary"], label="Trên trung bình"),
        Patch(facecolor=COLORS["primary"],   label="Dưới trung bình"),
    ]
    ax.legend(handles=legend_elements +
              [plt.Line2D([0], [0], color=COLORS["danger"],
                          lw=2, linestyle="--",
                          label=f"Mean = {mean_acc:.1f}% ± {np.std(fold_accs):.1f}%")],
              fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_metrics_bar(metrics: dict, save_path: str):
    """Sơ đồ 4: Bar chart các chỉ số Precision/Recall/F1/Specificity/Accuracy"""
    fig, ax = plt.subplots(figsize=(8, 5))

    names  = list(metrics.keys())
    values = list(metrics.values())
    colors_m = [
        COLORS["primary"],
        COLORS["danger"],
        COLORS["secondary"],
        COLORS["warning"],
        "#8B5CF6",
    ]

    bars = ax.barh(names, values, color=colors_m,
                   height=0.55, edgecolor="white", linewidth=1.5)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(min(val + 0.8, 99),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center", ha="left",
                fontsize=11, fontweight="bold")

    ax.set_xlim([0, 105])
    ax.set_xlabel("Giá trị (%)", labelpad=10)
    ax.set_title("Tổng hợp các chỉ số đánh giá mô hình\nTổng hợp 5 folds", pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Ngưỡng 95%
    ax.axvline(95, color=COLORS["gray"], lw=1.5,
               linestyle=":", label="Ngưỡng 95%")
    ax.legend(fontsize=9)

    # Emoji đánh giá
    for i, val in enumerate(values):
        icon = "✓" if val >= 95 else ("~" if val >= 85 else "✗")
        color = (COLORS["secondary"] if val >= 95
                 else (COLORS["warning"] if val >= 85 else COLORS["danger"]))
        ax.text(2, i, icon, va="center", fontsize=13,
                color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train_kfold():
    cfg        = load_config()
    num_frames = cfg["num_frames"]
    input_dim  = cfg["input_dim"]

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    print(f"\nDataset: X={X.shape}  y={y.shape}")
    print(f"  fall={int((y==1).sum())}  non_fall={int((y==0).sum())}")

    assert X.shape[1] == num_frames, f"num_frames mismatch: {X.shape[1]} vs {num_frames}"
    assert X.shape[2] == input_dim,  f"input_dim mismatch: {X.shape[2]} vs {input_dim}"

    X = X.astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_accs         = []
    fold_cms          = []
    all_labels_global = []
    all_probs_global  = []
    best_global_acc   = 0.0
    best_global_state = None
    best_fold         = 1

    print(f"\n{'='*50}")
    print(f"K-FOLD CROSS VALIDATION ({N_FOLDS} folds)")
    print(f"{'='*50}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_acc, best_state, val_dl = train_one_fold(
            X_train, y_train, X_val, y_val,
            input_dim, num_frames, device, fold)

        fold_accs.append(best_acc)

        # Evaluate best model của fold
        model = PoseTransformer(input_dim=input_dim, num_frames=num_frames).to(device)
        model.load_state_dict(best_state)
        model.eval()

        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb    = xb.to(device)
                logits = model(xb)
                probs  = torch.softmax(logits, dim=1)[:, 1]  # prob of fall
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(yb.numpy())
                all_probs.extend(probs.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fold_cms.append(cm)

        all_labels_global.extend(all_labels)
        all_probs_global.extend(all_probs)

        tn, fp, fn, tp = cm.ravel()
        print(f"  Fold {fold+1}: acc={best_acc:.1f}% | "
              f"TP={tp} TN={tn} FP={fp} FN={fn}")

        if best_acc > best_global_acc:
            best_global_acc   = best_acc
            best_global_state = best_state
            best_fold         = fold + 1

    # ── Tổng hợp ─────────────────────────────────────────────────────────────

    cm_total         = np.sum(fold_cms, axis=0)
    tn, fp, fn, tp   = cm_total.ravel()
    all_preds_global = (np.array(all_probs_global) >= 0.5).astype(int)

    accuracy    = (tp + tn) / (tp + tn + fp + fn) * 100
    precision   = precision_score(all_labels_global, all_preds_global) * 100
    recall      = recall_score(all_labels_global, all_preds_global) * 100
    f1          = f1_score(all_labels_global, all_preds_global) * 100
    specificity = tn / (tn + fp) * 100
    miss_rate   = fn / (fn + tp) * 100
    false_alarm = fp / (fp + tn) * 100
    mean_acc    = float(np.mean(fold_accs))
    std_acc     = float(np.std(fold_accs))

    print(f"\n{'='*50}")
    print(f"K-FOLD RESULTS")
    print(f"{'='*50}")
    for i, acc in enumerate(fold_accs):
        marker = " ← best" if i + 1 == best_fold else ""
        print(f"  Fold {i+1}: {acc:.1f}%{marker}")
    print(f"\n  Mean accuracy : {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  Precision     : {precision:.2f}%")
    print(f"  Recall        : {recall:.2f}%")
    print(f"  F1-Score      : {f1:.2f}%")
    print(f"  Specificity   : {specificity:.2f}%")
    print(f"  Miss Rate     : {miss_rate:.2f}%")
    print(f"  False Alarm   : {false_alarm:.2f}%")

    # ── Vẽ 4 sơ đồ ───────────────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"GENERATING FIGURES → {FIGURE_DIR}")
    print(f"{'='*50}")

    plot_confusion_matrix(
        cm_total,
        os.path.join(FIGURE_DIR, "1_confusion_matrix.png")
    )

    plot_roc_curve(
        all_labels_global,
        all_probs_global,
        os.path.join(FIGURE_DIR, "2_roc_curve.png")
    )

    plot_fold_accuracy(
        fold_accs,
        os.path.join(FIGURE_DIR, "3_fold_accuracy.png")
    )

    metrics = {
        "Accuracy":    accuracy,
        "Precision":   precision,
        "Recall\n(Sensitivity)": recall,
        "F1-Score":    f1,
        "Specificity": specificity,
    }
    plot_metrics_bar(
        metrics,
        os.path.join(FIGURE_DIR, "4_metrics_bar.png")
    )

    # ── Lưu model và results ──────────────────────────────────────────────────

    torch.save({
        "model_state": best_global_state,
        "val_acc":     best_global_acc,
        "fold":        best_fold,
        "kfold_mean":  mean_acc,
        "kfold_std":   std_acc,
        "config": {
            "input_dim":  input_dim,
            "num_frames": num_frames,
            "d_model":    D_MODEL,
            "nhead":      NHEAD,
            "num_layers": NUM_LAYERS,
            "dropout":    DROPOUT,
            "normalized": True,
            "labels":     {"0": "non_fall", "1": "fall"},
        }
    }, os.path.join(SAVE_DIR, "best_model.pth"))

    results = {
        "fold_accs":        fold_accs,
        "mean_acc":         mean_acc,
        "std_acc":          std_acc,
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "specificity":      specificity,
        "miss_rate":        miss_rate,
        "false_alarm_rate": false_alarm,
        "confusion_matrix": cm_total.tolist(),
    }
    with open(os.path.join(SAVE_DIR, "kfold_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Best model : {SAVE_DIR}/best_model.pth")
    print(f"✓ Results    : {SAVE_DIR}/kfold_results.json")
    print(f"✓ Figures    : {FIGURE_DIR}/")
    print(f"\n  1_confusion_matrix.png")
    print(f"  2_roc_curve.png")
    print(f"  3_fold_accuracy.png")
    print(f"  4_metrics_bar.png")


if __name__ == "__main__":
    train_kfold()