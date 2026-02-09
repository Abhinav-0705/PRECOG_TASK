#!/usr/bin/env python3
"""Bootstrap stats for Task1 fingerprints.

Usage:
  python3 scripts/task1_bootstrap_stats.py --class0 Class0 --class2 Class2 --out data/analysis --sample-size 5000 --n-bootstrap 300

Produces:
 - data/analysis/bootstrap_samples.csv  (per-sample metrics)
 - data/analysis/statistical_comparisons.csv  (pairwise tests)
 - data/analysis/boxplots_<metric>.png
 - data/analysis/summary_stats.json
"""
import argparse
import os
import random
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import spacy
from tqdm import trange
import textstat
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns


def read_texts_from_dir(path):
    texts = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.txt'):
                p = os.path.join(root, f)
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    texts.append(fh.read())
    return '\n\n'.join(texts)


def tokenize_words(text):
    # very simple whitespace split for indexing; spaCy used for sentences/syllables later
    words = [w for w in text.split() if w.strip()]
    return words


def cliff_delta(xs, ys):
    # compute Cliff's delta
    nx = len(xs)
    ny = len(ys)
    more = 0
    less = 0
    for x in xs:
        for y in ys:
            if x > y:
                more += 1
            elif x < y:
                less += 1
    return (more - less) / (nx * ny)


def compute_metrics_for_text(nlp, sample_text):
    doc = nlp(sample_text)
    n_words = sum(1 for t in doc if t.is_alpha)
    sents = list(doc.sents)
    n_sents = max(1, len(sents))
    asl = n_words / n_sents
    # syllables: use textstat.syllable_count per token
    syllables = 0
    for t in doc:
        if t.is_alpha:
            # textstat expects a string; for safety, lower and strip
            syllables += textstat.syllable_count(t.text)
    asw = syllables / n_words if n_words > 0 else 0
    # Flesch-Kincaid using formula
    fk = 0.39 * asl + 11.8 * asw - 15.59
    # SMOG and Gunning Fog
    try:
        smog = textstat.smog_index(sample_text)
    except Exception:
        smog = None
    try:
        gfog = textstat.gunning_fog(sample_text)
    except Exception:
        gfog = None
    return {
        'n_words': n_words,
        'n_sentences': n_sents,
        'asl': asl,
        'syllables': syllables,
        'asw': asw,
        'flesch_kincaid_grade': fk,
        'smog_index': smog,
        'gunning_fog': gfog,
    }


def make_sample_text_from_words(words, start_idx, sample_size):
    span = words[start_idx:start_idx+sample_size]
    return ' '.join(span)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class0', required=True, help='Path to Class0 dir (authors folders)')
    parser.add_argument('--class2', required=True, help='Path to Class2 dir (topic files)')
    parser.add_argument('--out', default='data/analysis', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=5000, help='Words per sample')
    parser.add_argument('--n-bootstrap', type=int, default=300, help='Bootstrap iterations per label')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = args.out
    ensure_dir(outdir)

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

    # Build label -> concatenated text mapping
    labels = {}
    # Class0: expecting subfolders (authors)
    for author in sorted(os.listdir(args.class0)):
        p = os.path.join(args.class0, author)
        if os.path.isdir(p):
            txt = read_texts_from_dir(p)
            labels[f'class0::{author}'] = txt
    # Class2: each .txt in the class2 dir is a label (topic file)
    for f in sorted(os.listdir(args.class2)):
        if f.lower().endswith('.txt'):
            label = f'class2::{os.path.splitext(f)[0]}'
            path = os.path.join(args.class2, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                labels[label] = fh.read()

    # For each label, create list of words
    label_words = {}
    for label, text in labels.items():
        words = tokenize_words(text)
        label_words[label] = words
        print(f'{label}: total words {len(words)}')

    # Per-sample records
    records = []

    for label, words in labels.items():
        wlist = label_words[label]
        total = len(wlist)
        sample_size = args.sample_size
        use_replacement = False
        if total < sample_size:
            # fallback to sampling with replacement of words
            use_replacement = True
            print(f'Warning: label {label} has only {total} words < sample_size {sample_size}; sampling with replacement')
        for i in trange(args.n_bootstrap, desc=f'Bootstrapping {label}'):
            if not use_replacement and total >= sample_size:
                start = random.randint(0, total - sample_size)
                sample_text = make_sample_text_from_words(wlist, start, sample_size)
            else:
                # sample words with replacement and join
                sampled = [random.choice(wlist) for _ in range(sample_size)] if total>0 else []
                sample_text = ' '.join(sampled)
            m = compute_metrics_for_text(nlp, sample_text)
            m.update({'label': label, 'bootstrap_id': i})
            records.append(m)

    df = pd.DataFrame.from_records(records)
    csv_out = os.path.join(outdir, 'bootstrap_samples.csv')
    df.to_csv(csv_out, index=False)
    print(f'Wrote per-sample metrics to {csv_out}')

    # Summaries and CIs
    summary = []
    metrics = ['asl', 'asw', 'flesch_kincaid_grade', 'smog_index', 'gunning_fog']
    for label, g in df.groupby('label'):
        rec = {'label': label, 'n_samples': len(g)}
        for m in metrics:
            vals = g[m].dropna().values
            rec[f'{m}_mean'] = float(np.mean(vals)) if len(vals)>0 else None
            rec[f'{m}_std'] = float(np.std(vals, ddof=1)) if len(vals)>1 else None
            if len(vals)>1:
                lo = float(np.percentile(vals, 2.5))
                hi = float(np.percentile(vals, 97.5))
            else:
                lo = hi = float(vals[0]) if len(vals)==1 else None
            rec[f'{m}_ci_lo'] = lo
            rec[f'{m}_ci_hi'] = hi
        summary.append(rec)
    summary_path = os.path.join(outdir, 'summary_stats.json')
    with open(summary_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Wrote summary stats to {summary_path}')

    # Pairwise comparisons: for each class0 author vs each class2 label
    comparisons = []
    class0_labels = [l for l in labels.keys() if l.startswith('class0::')]
    class2_labels = [l for l in labels.keys() if l.startswith('class2::')]
    for metric in ['asl', 'asw', 'flesch_kincaid_grade']:
        for a in class0_labels:
            for b in class2_labels:
                vals_a = df.loc[df['label']==a, metric].dropna().values
                vals_b = df.loc[df['label']==b, metric].dropna().values
                if len(vals_a)<2 or len(vals_b)<2:
                    stat = p = None
                    cld = None
                else:
                    stat, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                    cld = cliff_delta(vals_b, vals_a)  # positive means class2 > class0
                comparisons.append({'metric': metric, 'class0': a, 'class2': b, 'u_stat': stat, 'p_value': p, 'cliffs_delta': cld})
    comp_df = pd.DataFrame.from_records(comparisons)
    comp_path = os.path.join(outdir, 'statistical_comparisons.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f'Wrote pairwise comparisons to {comp_path}')

    # Plots: boxplots for each metric across labels
    plot_metrics = ['asl', 'asw', 'flesch_kincaid_grade']
    for m in plot_metrics:
        plt.figure(figsize=(10,6))
        sns.boxplot(x='label', y=m, data=df)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        ppath = os.path.join(outdir, f'boxplot_{m}.png')
        plt.savefig(ppath)
        plt.close()
        print(f'Wrote plot {ppath}')

    print('Done.')

if __name__ == '__main__':
    main()
