"""
evaluate_presets.py
-------------------
Đánh giá 3 preset threshold cố định trên toàn bộ dataset.
Dùng kết quả này để báo cáo luận văn.

Cách dùng:
    python evaluate_presets.py --dataset_dir train_tranformer/dataset
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn


class _PoseTransformer(nn.Module):
    def __init__(self, input_dim, num_frames, d_model=128, nhead=4,
                 num_layers=2, dropout=0.4, num_classes=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model),
            nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        self.pos_emb   = nn.Parameter(torch.randn(1, num_frames+1, d_model)*0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(32, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        return self.head(self.transformer(x)[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--batch_size",  type=int, default=64)
    args = ap.parse_args()
    d = args.dataset_dir

    # Paths
    X_path    = os.path.join(d, "X.npy")
    y_path    = os.path.join(d, "y.npy")
    ckpt_path = os.path.join(d, "checkpoints1", "best_model.pth")
    mean_path = os.path.join(d, "checkpoints1", "scaler_mean.npy")
    std_path  = os.path.join(d, "checkpoints1", "scaler_std.npy")

    for p in [X_path, y_path, ckpt_path, mean_path, std_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Không tìm thấy: {p}"); sys.exit(1)

    # Load + scale
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(int)
    mean = np.load(mean_path)
    std  = np.load(std_path)
    std  = np.where(std < 1e-6, 1.0, std)

    N, T, D  = X.shape
    X_scaled = ((X.reshape(-1, D) - mean) / std).reshape(N, T, D).astype(np.float32)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt["config"]
    model  = _PoseTransformer(
        input_dim=int(cfg["input_dim"]), num_frames=int(cfg["num_frames"]),
        d_model=int(cfg.get("d_model",128)), nhead=int(cfg.get("nhead",4)),
        num_layers=int(cfg.get("num_layers",2)), dropout=float(cfg.get("dropout",0.4)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    # Inference
    probs = []
    for i in range(0, N, args.batch_size):
        b = torch.tensor(X_scaled[i:i+args.batch_size]).to(device)
        with torch.no_grad():
            probs.append(torch.softmax(model(b), dim=1)[:, 1].cpu().numpy())
    p_fall = np.concatenate(probs)

    n_fall = (y == 1).sum()
    n_nf   = (y == 0).sum()

    # 3 preset cố định
    presets = {
        "Cẩn thận":  0.40,
        "Cân bằng":  0.60,
        "Chính xác": 0.80,
    }

    print("\n" + "="*72)
    print(f"  ĐÁNH GIÁ 3 PRESET — Fall Detection Transformer")
    print(f"  Dataset: {N} samples  |  fall={n_fall}  non_fall={n_nf}")
    print(f"  val_acc={ckpt.get('val_acc',0):.2f}%  "
          f"kfold_mean={ckpt.get('kfold_mean',0):.2f}%  "
          f"kfold_std={ckpt.get('kfold_std',0):.2f}%")
    print("="*72)

    results = {}
    for name, thr in presets.items():
        pred = (p_fall >= thr).astype(int)

        TP = int(((pred==1) & (y==1)).sum())
        FP = int(((pred==1) & (y==0)).sum())
        TN = int(((pred==0) & (y==0)).sum())
        FN = int(((pred==0) & (y==1)).sum())

        TPR  = TP / n_fall * 100   # Sensitivity / Recall
        FPR  = FP / n_nf   * 100   # False Alarm Rate
        TNR  = TN / n_nf   * 100   # Specificity
        FNR  = FN / n_fall * 100   # Miss Rate
        prec = TP / (TP+FP) * 100 if (TP+FP) > 0 else 0
        f1   = 2*TPR*prec/(TPR+prec) if (TPR+prec) > 0 else 0
        acc  = (TP+TN) / N * 100

        results[name] = {
            "threshold": thr,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "TPR": round(TPR, 2), "FPR": round(FPR, 2),
            "TNR": round(TNR, 2), "FNR": round(FNR, 2),
            "Precision": round(prec, 2),
            "F1": round(f1, 2),
            "Accuracy": round(acc, 2),
        }

        print(f"\n  ┌─ {name}  (threshold = {thr})")
        print(f"  │  Accuracy   : {acc:.2f}%")
        print(f"  │  Sensitivity: {TPR:.2f}%  "
              f"← bắt đúng {TP}/{n_fall} fall thật")
        print(f"  │  Specificity: {TNR:.2f}%  "
              f"← nhận đúng {TN}/{n_nf} non-fall")
        print(f"  │  Precision  : {prec:.2f}%")
        print(f"  │  F1-score   : {f1:.2f}%")
        print(f"  │  False Alarm: {FPR:.2f}%  "
              f"← báo nhầm {FP}/{n_nf} non-fall")
        print(f"  └─ Miss Rate  : {FNR:.2f}%  "
              f"← bỏ sót {FN}/{n_fall} fall thật")

    print("\n" + "="*72)
    print("  BẢNG TÓM TẮT (dùng cho luận văn):")
    print("="*72)
    print(f"  {'Preset':<12} {'Threshold':>10} {'Accuracy':>10} "
          f"{'Sensitivity':>12} {'Specificity':>12} "
          f"{'Precision':>10} {'F1':>8} {'FPR':>8}")
    print(f"  {'-'*84}")
    for name, r in results.items():
        print(f"  {name:<12} {r['threshold']:>10.2f} {r['Accuracy']:>9.2f}% "
              f"{r['TPR']:>11.2f}% {r['TNR']:>11.2f}% "
              f"{r['Precision']:>9.2f}% {r['F1']:>7.2f}% {r['FPR']:>7.2f}%")
    print("="*72)

    # Lưu JSON
    out_path = os.path.join(d, "preset_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": {
                "val_acc":     round(ckpt.get("val_acc", 0), 4),
                "kfold_mean":  round(ckpt.get("kfold_mean", 0), 4),
                "kfold_std":   round(ckpt.get("kfold_std", 0), 4),
                "num_frames":  cfg["num_frames"],
                "input_dim":   cfg["input_dim"],
            },
            "dataset": {"total": N, "fall": int(n_fall), "non_fall": int(n_nf)},
            "presets": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Lưu: {out_path}")

    # Vẽ bar chart
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        names  = list(results.keys())
        colors = ["#1D9E75", "#378ADD", "#BA7517"]

        # Chart 1: Sensitivity vs Specificity vs F1
        metrics1 = ["TPR", "TNR", "F1"]
        labels1  = ["Sensitivity\n(TPR)", "Specificity\n(TNR)", "F1-score"]
        x = np.arange(len(names))
        w = 0.25
        ax = axes[0]
        for i, (m, lbl) in enumerate(zip(metrics1, labels1)):
            vals = [results[n][m] for n in names]
            bars = ax.bar(x + i*w, vals, w, label=lbl)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + w)
        ax.set_xticklabels(names)
        ax.set_ylim([85, 103])
        ax.set_ylabel("Phần trăm (%)")
        ax.set_title("Sensitivity / Specificity / F1 theo preset")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Chart 2: FPR vs FNR (False Alarm vs Miss Rate)
        fpr_vals = [results[n]["FPR"] for n in names]
        fnr_vals = [results[n]["FNR"] for n in names]
        x2 = np.arange(len(names))
        w2 = 0.35
        ax2 = axes[1]
        b1 = ax2.bar(x2 - w2/2, fpr_vals, w2, label="False Alarm Rate (FPR)",
                     color="#E24B4A")
        b2 = ax2.bar(x2 + w2/2, fnr_vals, w2, label="Miss Rate (FNR)",
                     color="#888780")
        for bars in [b1, b2]:
            for bar in bars:
                v = bar.get_height()
                ax2.text(bar.get_x()+bar.get_width()/2, v+0.05,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(names)
        ax2.set_ylabel("Phần trăm (%)")
        ax2.set_title("False Alarm Rate vs Miss Rate theo preset")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Đánh giá 3 Preset — Transformer Fall Detection\n"
            f"Dataset: {N} samples | val_acc={ckpt.get('val_acc',0):.2f}% "
            f"| AUC≈0.997",
            fontsize=11
        )
        plt.tight_layout()
        chart_path = os.path.join(d, "preset_evaluation.png")
        plt.savefig(chart_path, dpi=150)
        print(f"[OK] Chart → {chart_path}")
        plt.show()
    except ImportError:
        print("[SKIP] pip install matplotlib để vẽ chart")


if __name__ == "__main__":
    main()