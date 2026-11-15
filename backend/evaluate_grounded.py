"""Offline evaluation: compare ML-only vs Source-only vs Fused credibility scoring.

Usage:
    python backend/evaluate_grounded.py data/ifnd.csv
    python backend/evaluate_grounded.py data/ifnd.csv --sample 500
"""
import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import joblib, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import retriever, similarity
from preprocess import clean
from explain import linguistic_markers

BUNDLE = joblib.load(ROOT / "models" / "model.joblib")
vec    = BUNDLE["vectorizer"]
model  = BUNDLE["model"]


def _ml_signal(text: str) -> float:
    return float(model.predict_proba(vec.transform([clean(text)]))[0][1])


def _fused_cred(p_real: float, source_signal: float,
                ling_score: float, confidence_flag: bool) -> float:
    ling_weight = min(0.60, ling_score * 0.8)
    ml_weight   = 1.0 - ling_weight
    ling_real   = 1.0 - ling_score
    p_real_adj  = ml_weight * p_real + ling_weight * ling_real
    if confidence_flag and ling_score <= 0.70:
        cred = 0.55 * p_real_adj + 0.45 * source_signal
    else:
        cred = p_real_adj
    return max(0.0, min(1.0, cred))


def _verdict(cred_0_1: float) -> int:
    score = int(round(cred_0_1 * 100))
    if score >= 60:  return 1
    if score < 40:   return 0
    return 0  # Unverified -> conservative: treat as fake


def _metrics(labels, preds, name: str) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0)
    return dict(
        config=name,
        accuracy=round(accuracy_score(labels, preds), 4),
        precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
        confusion=confusion_matrix(labels, preds, labels=[0, 1]).tolist())


def evaluate(csv_path: Path, sample) -> None:
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    rows = df.to_dict("records")
    if sample and sample < len(rows):
        random.seed(42)
        rows = random.sample(rows, sample)

    print(f"Evaluating {len(rows)} rows from {csv_path.name} ...")
    t_start = time.time()

    ml_preds, src_preds, fused_preds, true_labels = [], [], [], []
    fallback_count = 0

    for i, row in enumerate(rows, 1):
        text  = str(row["text"])
        label = int(row["label"])
        true_labels.append(label)

        p_real     = _ml_signal(text)
        ling_score = linguistic_markers(text)["score"]

        try:
            arts   = retriever.fetch(text)
            scores = similarity.compute_scores(text, arts)
            src_sig, conf_flag = similarity.compute_source_signal(arts, scores)
            if not arts:
                raise ValueError("empty corpus")
        except Exception:
            fallback_count += 1
            src_sig, conf_flag = 0.5, False

        ml_preds.append(_verdict(_fused_cred(p_real, 0.5, ling_score, False)))
        src_preds.append(1 if src_sig >= 0.5 else 0)
        fused_preds.append(_verdict(_fused_cred(p_real, src_sig, ling_score, conf_flag)))

        if i % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  {i}/{len(rows)} rows -- {elapsed:.1f}s elapsed -- fallbacks: {fallback_count}")

    results = [
        _metrics(true_labels, ml_preds,    "ml_only"),
        _metrics(true_labels, src_preds,   "source_only"),
        _metrics(true_labels, fused_preds, "fused"),
    ]

    out = dict(
        dataset=csv_path.stem, rows=len(df),
        sample=sample if sample else len(df),
        fallback_count=fallback_count,
        evaluated_at=time.strftime("%Y-%m-%d %H:%M"),
        results=results)

    out_path = ROOT / "models" / "grounded_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print(f"{'Config':<14}  {'Accuracy':>9}  {'Precision':>10}  {'Recall':>7}  {'F1':>7}")
    print("-" * 56)
    for r in results:
        print(f"{r['config']:<14}  {r['accuracy']:>9.4f}  {r['precision']:>10.4f}  "
              f"{r['recall']:>7.4f}  {r['f1']:>7.4f}")
    print()
    print(f"Fallback rows (live retrieval failed): {fallback_count} / {len(rows)}")
    print(f"Output written: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Offline evaluation for source-grounded verification.")
    ap.add_argument("csv_path", type=Path,
                    help="CSV file with columns: text, label (0=fake, 1=real)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Randomly sample N rows with seed=42 (default: all rows)")
    args = ap.parse_args()
    evaluate(args.csv_path, args.sample)
