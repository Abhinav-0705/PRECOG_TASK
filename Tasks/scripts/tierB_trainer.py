#!/usr/bin/env python3
"""scripts/tierB_trainer.py

Tier B (rebuilt from scratch): a lightweight *semantic-ish* detector that is fully
reproducible without any HuggingFace downloads at notebook time.

Design goals:
 - Works out-of-the-box with only scikit-learn installed.
 - Uses TF-IDF + linear classifier by default.
 - Optionally supports SentenceTransformer embeddings, but **only when explicitly enabled**.
 - Always saves artifacts for the notebook to display (incl. a confusion-matrix PNG):
     - data/analysis/tierB_dataset.csv
     - data/analysis/tierB_results.json
     - data/analysis/tierB_confusion.png
     - data/analysis/tierB_model.joblib

Run (from Task0/):
  python3 scripts/tierB_trainer.py --out data/analysis

Optional embeddings mode (may require downloads):
  python3 scripts/tierB_trainer.py --out data/analysis --repr embed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier B (clean): TF-IDF or embeddings")
    p.add_argument("--out", type=str, default="data/analysis", help="Output directory")
    p.add_argument(
        "--repr",
        type=str,
        default="tfidf",
        choices=["tfidf", "embed"],
        help="Representation: tfidf (default, no HF), embed (SentenceTransformer)",
    )
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument(
        "--max-paragraphs-per-class",
        type=int,
        default=0,
        help="Max paragraphs per class (0 = use all).",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def find_repo_root(start: Path) -> Path:
    """Find Task0/ by searching up from `start` for Class0/Class2/Class3."""
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
    paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return paras


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

        # deterministic truncation (keeps run reproducible)
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

    # Discover repo root from output folder (Task0/data/analysis -> Task0)
    repo_root = find_repo_root(out)
    if not (repo_root / "Class0").exists():
        # last resort: cwd
        repo_root = find_repo_root(Path.cwd())

    print("Repo root:", repo_root)
    texts, labels = build_dataset(repo_root, args.max_paragraphs_per_class)
    print("Built dataset paragraphs:", len(texts))

    # Save dataset snapshot (handy for debugging + reproducibility)
    try:
        import pandas as pd

        pd.DataFrame({"text": texts, "label": labels}).to_csv(out / "tierB_dataset.csv", index=False)
    except Exception:
        pass

    # Encode labels
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Build representation
    vectorizer = None
    embed_model_used = None

    if args.repr == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(texts)
    else:
        # Optional path — user explicitly asked for it.
        from sentence_transformers import SentenceTransformer

        # put cache *inside Task0* to avoid ~/.cache permission issues
        cache_dir = repo_root / ".hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(args.embed_model, cache_folder=str(cache_dir))
        X = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embed_model_used = args.embed_model

    # Split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    # Classifier
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=300, solver="saga" if hasattr(X_train, "tocoo") else "lbfgs")
    clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # CV (fast-ish)
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    # Confusion matrix + save figure
    cm = confusion_matrix(y_test, y_pred)
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
            ax=ax, cmap="Blues", values_format="d"
        )
        plt.title("Tier B confusion matrix")
        plt.tight_layout()
        fig.savefig(out / "tierB_confusion.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

    out_res = {
        "repr": args.repr,
        "embed_model": embed_model_used,
        "accuracy_test": acc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "classes": le.classes_.tolist(),
        "report": report,
    }
    (out / "tierB_results.json").write_text(json.dumps(out_res, indent=2), encoding="utf-8")

    # Save model bundle
    try:
        import joblib

        payload = {
            "model": clf,
            "le": le,
            "repr": args.repr,
            "embed_model": embed_model_used,
            "vectorizer": vectorizer,
        }
        joblib.dump(payload, out / "tierB_model.joblib")
    except Exception:
        pass

    print("Saved:")
    print(" -", out / "tierB_dataset.csv")
    print(" -", out / "tierB_results.json")
    print(" -", out / "tierB_confusion.png")
    print(" -", out / "tierB_model.joblib")


if __name__ == "__main__":
    main()
