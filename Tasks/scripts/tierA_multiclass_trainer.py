#!/usr/bin/env python3
"""Tier A multiclass trainer: human / neutral-ai / mimic-ai

Builds paragraph-level features from Class0 (human), Class2 (neutral AI), Class3 (mimic AI),
trains a RandomForest multiclass classifier, evaluates, and writes results to data/analysis.

Outputs:
- data/analysis/tierA_multiclass_results.json
- data/analysis/tierA_multiclass_confusion.png
- data/analysis/tierA_multiclass_feature_importances.csv
- data/analysis/tierA_multiclass_dataset.csv
"""
import argparse
from pathlib import Path
import re
from collections import Counter
import random
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def tokenize_words(text):
    return [w.lower() for w in re.findall(r"[A-Za-z']+", text) if any(c.isalpha() for c in w)]


def count_syllables(word):
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    vowels = 'aeiouy'
    sylls = 0
    prev = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev:
            sylls += 1
        prev = is_v
    if w.endswith('e'):
        sylls = max(1, sylls-1)
    return sylls


def fk_grade_from_text(text):
    words = tokenize_words(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    if not words or not sentences:
        return None, None, None
    ASL = len(words) / len(sentences)
    syllables = sum(count_syllables(w) for w in words)
    ASW = syllables / len(words)
    FK = 0.39 * ASL + 11.8 * ASW - 15.59
    return ASL, ASW, FK


PUNCT_TYPES = [';', '—', '–', '-', ':', ',', '.', '!', '?', '(', ')', '"', "'", '...']


def punct_features(text):
    c = Counter()
    for p in PUNCT_TYPES:
        c[p] = text.count(p)
    words = tokenize_words(text)
    nwords = max(1, len(words))
    per_1000 = {f'punct_per_1000_{re.sub(r"[^A-Za-z0-9]","_",p)}': c[p] / nwords * 1000 for p in c}
    per_1000['total_punct_per_1000'] = sum(c[p] for p in c) / nwords * 1000
    return per_1000


def build_paragraphs_from_folder(folder):
    p = Path(folder)
    paras = []
    if not p.exists():
        return paras
    for f in sorted(p.rglob('*.txt')):
        try:
            txt = f.read_text(encoding='utf-8')
        except Exception:
            txt = f.read_text(encoding='latin-1')
        blocks = [b.strip() for b in re.split(r'\n\s*\n', txt) if b.strip()]
        for b in blocks:
            paras.append({'source_file': str(f.relative_to(p)), 'text': b})
    return paras


def extract_features_from_text(text):
    words = tokenize_words(text)
    total = len(words)
    uniq = len(set(words))
    ttr = uniq / total if total > 0 else 0
    counts = Counter(words)
    hapax = sum(1 for w,c in counts.items() if c==1)
    ASL, ASW, FK = fk_grade_from_text(text)
    pf = punct_features(text)
    feats = {
        'words': total,
        'unique_words': uniq,
        'ttr': ttr,
        'hapax': hapax,
        'ASL': ASL,
        'ASW': ASW,
        'FK': FK,
    }
    feats.update(pf)
    return feats


def build_dataset(class0_dir, class2_dir, class3_dir, max_per_class=2000, random_state=1):
    paras_h = build_paragraphs_from_folder(class0_dir)
    paras_n = build_paragraphs_from_folder(class2_dir)
    paras_a = build_paragraphs_from_folder(class3_dir)
    random.seed(random_state)
    if len(paras_h) > max_per_class:
        paras_h = random.sample(paras_h, max_per_class)
    if len(paras_n) > max_per_class:
        paras_n = random.sample(paras_n, max_per_class)
    if len(paras_a) > max_per_class:
        paras_a = random.sample(paras_a, max_per_class)
    rows = []
    for p in paras_h:
        feats = extract_features_from_text(p['text'])
        feats['label'] = 'human'
        rows.append(feats)
    for p in paras_n:
        feats = extract_features_from_text(p['text'])
        feats['label'] = 'neutral_ai'
        rows.append(feats)
    for p in paras_a:
        feats = extract_features_from_text(p['text'])
        feats['label'] = 'mimic_ai'
        rows.append(feats)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=['FK'])
    return df


def train_and_eval(df, out_dir):
    X = df.drop(columns=['label'])
    y = df['label']
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=['human','neutral_ai','mimic_ai'])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    fi = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_dir / 'tierA_multiclass_dataset.parquet', index=False)
    except Exception:
        df.to_csv(out_dir / 'tierA_multiclass_dataset.csv', index=False)
    with open(out_dir / 'tierA_multiclass_results.json', 'w', encoding='utf-8') as fh:
        json.dump({'accuracy_test': acc, 'cv_mean': float(cv_scores.mean()), 'cv_std': float(cv_scores.std()), 'report': report}, fh, indent=2)
    fi.to_csv(out_dir / 'tierA_multiclass_feature_importances.csv', index=False)
    # confusion matrix plot
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Multiclass confusion matrix (test)')
    plt.colorbar()
    plt.xticks([0,1,2], ['human','neutral_ai','mimic_ai'])
    plt.yticks([0,1,2], ['human','neutral_ai','mimic_ai'])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(out_dir / 'tierA_multiclass_confusion.png')
    print('Saved multiclass results to', out_dir)
    print('Test accuracy:', acc)
    print('CV mean/std:', cv_scores.mean(), cv_scores.std())
    print('Top features:')
    print(fi.head(10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class0', default='Class0')
    parser.add_argument('--class2', default='Class2')
    parser.add_argument('--class3', default='Class3')
    parser.add_argument('--out', default='data/analysis')
    parser.add_argument('--max-per-class', type=int, default=2000)
    args = parser.parse_args()
    df = build_dataset(args.class0, args.class2, args.class3, max_per_class=args.max_per_class)
    print('Built dataset rows=', len(df))
    train_and_eval(df, args.out)


if __name__ == '__main__':
    main()
