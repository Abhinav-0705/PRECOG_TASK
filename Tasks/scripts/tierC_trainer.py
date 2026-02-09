#!/usr/bin/env python3
"""scripts/tierC_trainer.py

Tier C (lightweight, from scratch): character n-gram detector.

Why this for Tier C?
- You asked to *train/test/save results into the notebook*, but your environment has
  HuggingFace permission issues (token/cache). A full Transformer/LoRA pipeline would
  likely fail or be too heavy.
- A char n-gram model is a strong "deep-ish" baseline for stylistic signals and often
  outperforms word TF-IDF on authorship/AI mimic tasks.

Artifacts written to --out (default data/analysis):
- tierC_dataset.csv
- tierC_results.json
- tierC_confusion.png
- tierC_model.joblib

Run (from Task0/):
  python3 scripts/tierC_trainer.py --out data/analysis

Notes:
- Uses only scikit-learn + matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier C: char n-gram detector")
    p.add_argument("--out", type=str, default="data/analysis")
    p.add_argument(
        "--max-paragraphs-per-class",
        type=int,
        default=0,
        help="Max paragraphs per class (0 = use all).",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-seed", type=int, default=42)
    # char n-grams are the 'tier C' knob
    p.add_argument("--char-ngram-min", type=int, default=3)
    p.add_argument("--char-ngram-max", type=int, default=5)
    p.add_argument("--max-features", type=int, default=80000)
    return p.parse_args()


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(6):
        if (cur / "Class0").exists() and (cur / "Class2").exists() and (cur / "Class3").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def read_paragraphs(txt_path: Path) -> List[str]:
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    return [p.strip() for p in raw.split("\n\n") if p.strip()]


def build_dataset(repo_root: Path, max_paragraphs_per_class: int) -> Tuple[List[str], List[str]]:
    mapping = [("Class0", "human"), ("Class2", "ai_neutral"), ("Class3", "ai_mimic")]

    texts: List[str] = []
    labels: List[str] = []

    for folder_name, label in mapping:
        folder = repo_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        class_paras: List[str] = []
        for p in sorted(folder.rglob("*.txt")):
            try:
                class_paras.extend(read_paragraphs(p))
            except Exception:
                continue

        if max_paragraphs_per_class and max_paragraphs_per_class > 0:
            class_paras = class_paras[:max_paragraphs_per_class]
        texts.extend(class_paras)
        labels.extend([label] * len(class_paras))

    if not texts:
        raise RuntimeError("No paragraphs found under Class0/Class2/Class3")

    return texts, labels


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    repo_root = find_repo_root(out)
    if not (repo_root / "Class0").exists():
        repo_root = find_repo_root(Path.cwd())

    print("Repo root:", repo_root)
    texts, labels = build_dataset(repo_root, args.max_paragraphs_per_class)
    print("Built dataset paragraphs:", len(texts))

    # Save dataset snapshot
    try:
        import pandas as pd

        pd.DataFrame({"text": texts, "label": labels}).to_csv(out / "tierC_dataset.csv", index=False)
    except Exception:
        pass

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.char_ngram_min, args.char_ngram_max),
        max_features=args.max_features,
        min_df=2,
    )
    X = vec.fit_transform(texts)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=400, solver="saga")
    clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    # Save confusion plot
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
            ax=ax, cmap="Blues", values_format="d"
        )
        plt.title("Tier C confusion matrix (char n-grams)")
        plt.tight_layout()
        fig.savefig(out / "tierC_confusion.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

    out_res = {
        "method": "char_tfidf",
        "char_ngram_range": [args.char_ngram_min, args.char_ngram_max],
        "max_features": args.max_features,
        "accuracy_test": acc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "classes": le.classes_.tolist(),
        "report": report,
    }
    (out / "tierC_results.json").write_text(json.dumps(out_res, indent=2), encoding="utf-8")

    # Save model bundle
    try:
        import joblib

        payload = {"model": clf, "le": le, "vectorizer": vec, "method": out_res["method"]}
        joblib.dump(payload, out / "tierC_model.joblib")
    except Exception:
        pass

    print("Saved:")
    print(" -", out / "tierC_dataset.csv")
    print(" -", out / "tierC_results.json")
    print(" -", out / "tierC_confusion.png")
    print(" -", out / "tierC_model.joblib")


if __name__ == "__main__":
    main()
