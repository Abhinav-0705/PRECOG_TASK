#!/usr/bin/env python3
"""Bootstrap-based comparisons for Task1 fingerprint metrics.

Reads text folders for `class0` (author subfolders) and `class2` (topic subfolders),
draws bootstrap samples of words, computes metrics per sample (TTR, hapax, adj/noun,
avg dependency depth, punctuation per 1000 words, Flesch-Kincaid grade), and runs
pairwise Mann-Whitney U tests + Cliff's delta between groups. Saves results to
`--out` as CSV, JSON and boxplots.

Example:
  python3 scripts/task1_stats.py --class0 Class0 --class2 Class2 --out data/analysis --sample-size 5000 --n-boot 300
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textstat
import spacy
from scipy.stats import mannwhitneyu


def read_all_words(folder):
    texts = []
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if fn.lower().endswith('.txt'):
                path = os.path.join(root, fn)
                with open(path, 'r', encoding='utf8') as f:
                    texts.append(f.read())
    joined = '\n'.join(texts)
    # simple word tokenizer (keeps apostrophes inside words)
    words = re.findall(r"\b[\w']+\b", joined)
    return words, joined


def sample_words(words, n):
    if len(words) == 0:
        return []
    if len(words) >= n:
        return random.sample(words, n)
    # sample with replacement if insufficient words
    return [random.choice(words) for _ in range(n)]


def compute_metrics_from_text(sample_text, nlp):
    words = re.findall(r"\b[\w']+\b", sample_text)
    total = len(words)
    unique = len(set(w.lower() for w in words))
    ttr = unique / total if total > 0 else 0.0
    hapax = sum(1 for _, c in Counter(w.lower() for w in words).items() if c == 1)

    doc = nlp(sample_text)
    adj = sum(1 for t in doc if t.pos_ == 'ADJ')
    noun = sum(1 for t in doc if t.pos_ == 'NOUN')
    adj_noun_ratio = adj / noun if noun > 0 else float('nan')

    # average dependency tree depth (token -> root distance)
    depths = []
    for t in doc:
        if t.is_punct:
            continue
        d = 0
        cur = t
        while cur.head is not cur:
            d += 1
            cur = cur.head
            if d > 100:
                break
        depths.append(d)
    avg_dep_depth = float(np.mean(depths)) if depths else float('nan')

    # punctuation counts
    punct_chars = re.findall(r"[;\u2014\u2013\-:,\.\!\?()\"'…]", sample_text)
    total_punct = len(punct_chars)
    punct_per_1000 = total_punct / total * 1000 if total > 0 else 0.0

    fk = textstat.flesch_kincaid_grade(sample_text)

    return {
        'ttr': ttr,
        'hapax': hapax,
        'adj_noun_ratio': adj_noun_ratio,
        'avg_dep_depth': avg_dep_depth,
        'punct_per_1000': punct_per_1000,
        'fk_grade': fk,
    }


def cliffs_delta(a, b):
    # compute Cliff's delta
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('nan')
    more = 0
    less = 0
    for x in a:
        for y in b:
            if x > y:
                more += 1
            elif x < y:
                less += 1
    return (more - less) / (n * m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class0', required=True, help='Path to class0 folder (authors as subfolders)')
    parser.add_argument('--class2', required=True, help='Path to class2 folder (topics as subfolders)')
    parser.add_argument('--out', required=True, help='Output directory for statistics')
    parser.add_argument('--sample-size', type=int, default=5000, help='Words per bootstrap sample')
    parser.add_argument('--n-boot', type=int, default=300, help='Number of bootstrap samples per entity')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Loading spaCy model (this may take a moment)...')
    nlp = spacy.load(args.spacy_model, disable=['ner'])

    entities = {}

    # load authors
    for name in sorted(os.listdir(args.class0)):
        path = os.path.join(args.class0, name)
        if os.path.isdir(path):
            words, joined = read_all_words(path)
            entities[f'class0::{name}'] = {'words': words, 'text': joined}

    # load class2 topics
    for name in sorted(os.listdir(args.class2)):
        path = os.path.join(args.class2, name)
        if os.path.isdir(path):
            words, joined = read_all_words(path)
            entities[f'class2::{name}'] = {'words': words, 'text': joined}

    results = {}
    rnd = random.Random(1234)

    for label, data in entities.items():
        words = data['words']
        joined = data['text']
        print(f'Bootstrapping {label} (total words={len(words)})')
        arrs = defaultdict(list)
        for i in range(args.n_boot):
            sample_w = sample_words(words, args.sample_size)
            sample_text = ' '.join(sample_w)
            metrics = compute_metrics_from_text(sample_text, nlp)
            for k, v in metrics.items():
                arrs[k].append(v)
        results[label] = {k: list(v) for k, v in arrs.items()}

    # save bootstrap distributions
    with open(os.path.join(args.out, 'bootstrap_distributions.json'), 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2)

    # For each metric, run pairwise comparisons between class0 authors and each class2 topic
    metrics = ['ttr', 'hapax', 'adj_noun_ratio', 'avg_dep_depth', 'punct_per_1000', 'fk_grade']
    rows = []
    labels = sorted(results.keys())
    class0_labels = [l for l in labels if l.startswith('class0::')]
    class2_labels = [l for l in labels if l.startswith('class2::')]

    for metric in metrics:
        for a in class0_labels:
            for b in class2_labels:
                A = np.array(results[a][metric])
                B = np.array(results[b][metric])
                # Mann-Whitney U
                try:
                    u, p = mannwhitneyu(A, B, alternative='two-sided')
                except Exception:
                    u, p = float('nan'), float('nan')
                cd = cliffs_delta(A, B)
                rows.append({
                    'metric': metric,
                    'group_a': a,
                    'group_b': b,
                    'median_a': float(np.median(A)),
                    'median_b': float(np.median(B)),
                    'u_stat': float(u),
                    'p_value': float(p),
                    'cliffs_delta': float(cd),
                    'mean_diff': float(np.mean(A) - np.mean(B))
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'pairwise_comparisons.csv'), index=False)

    # Plot boxplots per metric comparing class0 vs class2
    for metric in metrics:
        rows = []
        for label in labels:
            vals = results[label][metric]
            for v in vals:
                rows.append({'label': label, 'metric': metric, 'value': v})
        pdf = pd.DataFrame(rows)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='label', y='value', data=pdf)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Bootstrap distributions: {metric}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'boxplot_{metric}.png'))
        plt.close()

    print('Bootstrap distributions saved to', os.path.join(args.out, 'bootstrap_distributions.json'))
    print('Pairwise comparisons saved to', os.path.join(args.out, 'pairwise_comparisons.csv'))
    print('Boxplots saved to', args.out)


if __name__ == '__main__':
    main()
