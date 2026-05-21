"""
train_transformer_kfold.py
Train Transformer với K-Fold Cross Validation
→ Đánh giá tin cậy hơn với dataset nhỏ

Thay đổi so với version cũ:
  - Bỏ StandardScaler (extract_keypoints đã normalize rồi)
  - Sửa augmentation flip lỗi numpy indexing
  - Thêm check balance sau khi load data
  - Thêm warning nếu mất balance
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import os
import json
import copy

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR     = r"D:\luan_van\Fall-Detection\train_tranformer\dataset_1"
SAVE_DIR     = r"D:\luan_van\Fall-Detection\train_tranformer\dataset_1\checkpoints1"

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

# ── Load config ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config loaded:")
        print(f"  num_frames  = {cfg['num_frames']}")
        print(f"  input_dim   = {cfg['input_dim']}")
        print(f"  normalized  = {cfg.get('normalized', False)}")
        print(f"  window_sec  = {cfg.get('window_sec', 'N/A')}")
        if cfg.get("fusion"):
            print(f"  fusion      = MediaPipe({cfg.get('mp_kp',33)} kp) "
                  f"+ YOLO({cfg.get('yolo_kp',17)} kp)")
        sampling = cfg.get("sampling", {})
        if sampling:
            print(f"  sampling    = fall:{sampling.get('fall','?')} | "
                  f"non_fall:{sampling.get('non_fall','?')}")
        return cfg
    else:
        print("WARNING: config.json không tìm thấy, dùng default")
        return {"num_frames": 30, "input_dim": 200, "fusion": True,
                "normalized": True}

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
    """
    Augment training data để tăng số lượng sample
    Chỉ apply trên train set, không apply trên val set
    """
    X_aug, y_aug = [X], [y]

    for _ in range(n_aug):
        X_new = X.copy()

        # 1. Horizontal flip — lật trái phải keypoints
        #    x_new = 1 - x (chỉ flip tọa độ x, index 0,4,8,... trong vector 200)
        mask     = np.random.rand(len(X_new)) > 0.5
        mask_idx = np.where(mask)[0]
        if len(mask_idx) > 0:
            X_flip   = X_new.copy()
            x_indices = np.arange(0, input_dim, 4)  # index của tọa độ x
            # Sửa lỗi: dùng np.ix_ thay vì chaining index
            idx = np.ix_(mask_idx,
                         range(X_new.shape[1]),
                         x_indices.tolist())
            X_flip[idx] = 1.0 - X_flip[idx]
            X_new = X_flip

        # 2. Speed change — thay đổi tốc độ ±20%
        factor    = np.random.choice([0.8, 0.9, 1.1, 1.2])
        T         = X_new.shape[1]
        src_len   = max(1, int(T * factor))
        src_idx   = np.linspace(0, T - 1, src_len)
        src_idx   = np.clip(src_idx, 0, T - 1).astype(int)
        final_idx = np.linspace(0, len(src_idx) - 1, T, dtype=int)
        X_new     = X_new[:, src_idx, :][:, final_idx, :]

        # 3. Gaussian noise nhỏ — simulate detection noise
        X_new = X_new + np.random.normal(0, 0.01, X_new.shape)

        # 4. Random frame dropout — simulate frame miss
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
        self.pos_emb   = nn.Parameter(
            torch.randn(1, num_frames + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
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
    """Train 1 fold → trả về best val_acc, best model state, val_dl"""

    # Augment train only — val giữ nguyên không augment
    X_tr, y_tr = augment(X_train, y_train, input_dim=input_dim, n_aug=3)
    print(f"  Train sau augment: {len(X_tr)} samples "
          f"(fall={int((y_tr==1).sum())} non_fall={int((y_tr==0).sum())})")

    train_dl = DataLoader(
        PoseDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(
        PoseDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PoseTransformer(
        input_dim=input_dim, num_frames=num_frames).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    print(f"\n  {'Ep':>4} {'Loss':>7} {'Tr%':>7} {'Val%':>7}")
    print(f"  {'-'*30}")

    for epoch in range(1, EPOCHS + 1):
        # Train
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

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_correct += model(xb).argmax(1).eq(yb).sum().item()

        val_acc = val_correct / len(val_dl.dataset) * 100

        marker = " ←" if val_acc > best_val_acc else ""
        # In mỗi 5 epoch hoặc khi có cải thiện
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

# ── K-Fold train ──────────────────────────────────────────────────────────────

def train_kfold():
    # Bước 1: Load config từ extract_keypoints
    cfg        = load_config()
    num_frames = cfg["num_frames"]
    input_dim  = cfg["input_dim"]

    # Kiểm tra đã normalize chưa
    if not cfg.get("normalized", False):
        print("WARNING: Data chưa được normalize trong extract_keypoints!")
        print("         Kiểm tra lại file extract_keypoints.py")

    # Bước 2: Load data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    N, T, D = X.shape

    print(f"\nDataset loaded: X={X.shape}  y={y.shape}")
    n_fall = int((y == 1).sum())
    n_nf   = int((y == 0).sum())
    print(f"  fall={n_fall}  non_fall={n_nf}")

    # Kiểm tra shape khớp config
    assert T == num_frames, \
        f"num_frames mismatch: data={T} config={num_frames}"
    assert D == input_dim, \
        f"input_dim mismatch: data={D} config={input_dim}"

    # Kiểm tra balance
    diff = abs(n_fall - n_nf)
    if diff > 30:
        print(f"\nWARNING: Mất balance! fall={n_fall} vs non_fall={n_nf} "
              f"(chênh {diff} video)")
        print("         Cân nhắc kiểm tra lại video bị skip trong extract")
    else:
        print(f"  Balance OK (chênh {diff} video)")

    # KHÔNG normalize lại — extract_keypoints đã normalize per-sequence
    # Chỉ đảm bảo dtype đúng
    X = X.astype(np.float32)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Bước 3: K-Fold
    kfold             = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                        random_state=42)
    fold_accs         = []
    fold_cms          = []
    best_global_acc   = 0.0
    best_global_state = None
    best_fold         = 1

    print(f"\n{'='*50}")
    print(f"K-FOLD CROSS VALIDATION ({N_FOLDS} folds)")
    print(f"{'='*50}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──")
        print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_acc, best_state, val_dl = train_one_fold(
            X_train, y_train, X_val, y_val,
            input_dim, num_frames, device, fold)

        fold_accs.append(best_acc)
        print(f"  Fold {fold+1} best val: {best_acc:.1f}%")

        # Confusion matrix của fold này
        model = PoseTransformer(
            input_dim=input_dim, num_frames=num_frames).to(device)
        model.load_state_dict(best_state)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                all_preds.extend(model(xb).argmax(1).cpu().numpy())
                all_labels.extend(yb.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fold_cms.append(cm)
        print(f"  Confusion matrix fold {fold+1}:")
        print(f"    non_fall: TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    fall    : FN={cm[1,0]}  TP={cm[1,1]}")

        # Lưu model tốt nhất toàn bộ
        if best_acc > best_global_acc:
            best_global_acc   = best_acc
            best_global_state = best_state
            best_fold         = fold + 1

    # Bước 4: Kết quả tổng hợp
    print(f"\n{'='*50}")
    print(f"K-FOLD RESULTS")
    print(f"{'='*50}")
    for i, acc in enumerate(fold_accs):
        marker = " ← best" if i + 1 == best_fold else ""
        print(f"  Fold {i+1}: {acc:.1f}%{marker}")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    print(f"\n  Mean accuracy : {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  Best fold     : Fold {best_fold} ({best_global_acc:.1f}%)")

    # Tổng hợp confusion matrix
    cm_total = np.sum(fold_cms, axis=0)
    tn, fp, fn, tp = cm_total.ravel()
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    false_alarm = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    miss_rate   = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

    print(f"\n  Tổng hợp {N_FOLDS} folds:")
    print(f"  Sensitivity (fall recall)  : {sensitivity:.1f}%")
    print(f"  Specificity (non_fall rec) : {specificity:.1f}%")
    print(f"  False Alarm Rate           : {false_alarm:.1f}%")
    print(f"  Miss Rate (bỏ sót té ngã)  : {miss_rate:.1f}%")

    print(f"\n  Confusion Matrix (tổng {N_FOLDS} folds):")
    print(f"                Pred non_fall  Pred fall")
    print(f"  True non_fall:    {tn:>6}       {fp:>6}")
    print(f"  True fall:        {fn:>6}       {tp:>6}")

    # Bước 5: Lưu best model
    torch.save({
        "model_state": best_global_state,
        "val_acc":     best_global_acc,
        "fold":        best_fold,
        "kfold_mean":  mean_acc,
        "kfold_std":   std_acc,
        # Lưu config để inference dùng — KHÔNG cần scaler vì đã normalize
        "config": {
            "input_dim":   input_dim,
            "num_frames":  num_frames,
            "d_model":     D_MODEL,
            "nhead":       NHEAD,
            "num_layers":  NUM_LAYERS,
            "dropout":     DROPOUT,
            "normalized":  True,   # extract_keypoints đã normalize
            "labels":      {"0": "non_fall", "1": "fall"},
        }
    }, os.path.join(SAVE_DIR, "best_model.pth"))

    # Lưu kết quả
    results = {
        "fold_accs":   fold_accs,
        "mean_acc":    mean_acc,
        "std_acc":     std_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_alarm": false_alarm,
        "miss_rate":   miss_rate,
        "confusion_matrix": cm_total.tolist(),
    }
    with open(os.path.join(SAVE_DIR, "kfold_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest model saved : {SAVE_DIR}/best_model.pth")
    print(f"Results saved    : {SAVE_DIR}/kfold_results.json")

    # Đánh giá tổng thể
    print(f"\n{'='*50}")
    if mean_acc >= 95:
        print(f"✓ Model TỐT      — Mean={mean_acc:.1f}% ± {std_acc:.1f}%")
    elif mean_acc >= 85:
        print(f"~ Model TRUNG BÌNH — Mean={mean_acc:.1f}% ± {std_acc:.1f}%")
    else:
        print(f"✗ Model YẾU      — Mean={mean_acc:.1f}% ± {std_acc:.1f}%")

    if std_acc <= 3:
        print(f"✓ Model ỔN ĐỊNH  — std={std_acc:.1f}% (ít variance giữa các fold)")
    else:
        print(f"✗ Model KHÔNG ỔN ĐỊNH — std={std_acc:.1f}% (cần thêm data)")

    if miss_rate <= 5:
        print(f"✓ Miss Rate tốt  — {miss_rate:.1f}% bỏ sót té ngã")
    else:
        print(f"✗ Miss Rate cao  — {miss_rate:.1f}% bỏ sót té ngã (nguy hiểm!)")


if __name__ == "__main__":
    train_kfold()