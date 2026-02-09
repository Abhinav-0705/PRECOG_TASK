"""
Task1 fingerprint analysis script

Computes per-author and per-class metrics:
- Type-Token Ratio (TTR) on 5,000-word samples
- Hapax Legomena count in 5,000-word samples
- POS distribution (Adj / Noun ratio)
- Average dependency tree depth per sentence (SpaCy)
- Punctuation counts & density heatmap
- Readability: Flesch-Kincaid Grade Level (textstat)

Outputs:
- data/analysis/task1_summary.json
- data/analysis/punct_heatmap.png
- data/analysis/metrics_table.csv

Usage:
    python3 scripts/task1_fingerprint.py --class0 data/class0 --class2 data/Class2 --out data/analysis

Note: this script requires spaCy and a model (en_core_web_sm) and textstat. Install with:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

"""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import random
import re
import math

import pandas as pd
import numpy as np

# NLP (optional)
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except Exception:
        nlp = None
except Exception:
    spacy = None
    nlp = None

# readability
try:
    import textstat
except Exception:
    textstat = None

PUNCT_TYPES = [';', '—', '–', '-', ':', ',', '.', '!', '?', '(', ')', '"', "'", '...']


def read_texts_from_dir(dirpath):
    p = Path(dirpath)
    texts = []
    for txt in sorted(p.rglob('*.txt')):
        try:
            t = txt.read_text(encoding='utf-8')
        except Exception:
            t = txt.read_text(encoding='latin-1')
        texts.append({'path': str(txt.relative_to(p)), 'text': t})
    return texts


def tokenize_words(text):
    # simple word tokenizer: alphabetic tokens including apostrophe contractions
    words = re.findall(r"[A-Za-z']+", text)
    return [w.lower() for w in words if any(c.isalpha() for c in w)]


def ttr_and_hapax(words, sample_size=5000, seed=0):
    if len(words) < sample_size:
        sample = words
    else:
        random.seed(seed)
        start = random.randint(0, max(0, len(words) - sample_size))
        sample = words[start:start+sample_size]
    total = len(sample)
    types = set(sample)
    ttr = len(types) / total if total>0 else 0
    counts = Counter(sample)
    hapax = sum(1 for w,c in counts.items() if c==1)
    return {'total_words': total, 'unique_words': len(types), 'ttr': ttr, 'hapax': hapax}


def pos_adj_noun_ratio(doc):
    # doc: spaCy doc
    adj = sum(1 for t in doc if t.pos_ == 'ADJ')
    noun = sum(1 for t in doc if t.pos_ in ('NOUN','PROPN'))
    ratio = (adj / noun) if noun>0 else float('nan')
    return {'adj': adj, 'noun': noun, 'adj_noun_ratio': ratio}


def dep_tree_depth(sent):
    # compute max depth of dependency tree for sentence (spaCy span)
    # find root and compute depth recursively
    def node_depth(token):
        if not list(token.children):
            return 1
        return 1 + max(node_depth(child) for child in token.children)
    try:
        root = [t for t in sent if t.head == t][0]
    except Exception:
        # fallback: pick first token
        root = sent[0]
    return node_depth(root)


def punct_counts(text):
    c = Counter()
    for p in PUNCT_TYPES:
        c[p] = text.count(p)
    # also total punctuation
    c['total_punct'] = sum(c[p] for p in PUNCT_TYPES)
    # length normalized per 1000 words approx
    words = tokenize_words(text)
    per_1000 = {p: (c[p] / max(1, len(words)) * 1000) for p in c}
    return c, per_1000


def flesch_kincaid_grade(text):
    if textstat is None:
        return None
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return None


def analyze_corpus(text_items, label, out_dir, sample_size=5000):
    # text_items: list of {'path','text'}
    joined = '\n'.join(t['text'] for t in text_items)
    words = tokenize_words(joined)
    ttr_stats = ttr_and_hapax(words, sample_size=sample_size)

    # spaCy processing: sentences
    global nlp
    pos_counts = {'adj':0,'noun':0}
    depths = []
    adj_noun_ratios = []
    if nlp is None:
        print('spaCy not loaded; skipping POS and dependency analyses')
    else:
        doc = nlp(joined)
        # pos distribution
        for sent in doc.sents:
            pr = pos_adj_noun_ratio(sent)
            adj_noun_ratios.append(pr['adj_noun_ratio'])
            pos_counts['adj'] += pr['adj']
            pos_counts['noun'] += pr['noun']
            try:
                d = dep_tree_depth(list(sent))
                depths.append(d)
            except Exception:
                pass
    avg_adj_noun = np.nanmean(adj_noun_ratios) if adj_noun_ratios else None
    avg_depth = float(np.mean(depths)) if depths else None

    # punctuation
    c, per_1000 = punct_counts(joined)

    # readability
    fk = flesch_kincaid_grade(joined)

    result = {
        'label': label,
        'n_docs': len(text_items),
        'words_total': len(words),
        'ttr_sample': ttr_stats,
        'pos_counts': pos_counts,
        'avg_adj_noun_ratio': avg_adj_noun,
        'avg_dep_tree_depth': avg_depth,
        'punct_counts': c,
        'punct_per_1000': per_1000,
        'flesch_kincaid_grade': fk
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class0', default='data/class0', help='Path to class0 (human) texts')
    parser.add_argument('--class2', default='data/Class2', help='Path to class2 (neutral generated) texts')
    parser.add_argument('--class3', default=None, help='Path to class3 (AI-mimic) texts (optional)')
    parser.add_argument('--out', default='data/analysis', help='Output folder')
    parser.add_argument('--sample-size', type=int, default=5000, help='word sample size for TTR/hapax')
    args = parser.parse_args()

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

    # load class0 authors (assume structure data/class0/{author}/*)
    class0_dir = Path(args.class0)
    corpora = []
    if class0_dir.exists():
        for author_dir in sorted(class0_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            items = read_texts_from_dir(author_dir)
            corpora.append((f'class0::{author_dir.name}', items))
    else:
        print('class0 dir not found:', class0_dir)

    # class2 (neutral) — treat each topic file as one corpus
    class2_dir = Path(args.class2)
    if class2_dir.exists():
        # each .txt file in class2 is a topic file → corpus
        for txt in sorted(class2_dir.glob('*.txt')):
            try:
                t = txt.read_text(encoding='utf-8')
            except Exception:
                t = txt.read_text(encoding='latin-1')
            corpora.append((f'class2::{txt.stem}', [{'path': txt.name, 'text': t}]))
    else:
        print('class2 dir not found:', class2_dir)

    # class3 (AI mimic) — optional; expect structure data/Class3/{author}/*.txt or similar
    if args.class3:
        class3_dir = Path(args.class3)
        if class3_dir.exists():
            for author_dir in sorted(class3_dir.iterdir()):
                if not author_dir.is_dir():
                    continue
                items = read_texts_from_dir(author_dir)
                corpora.append((f'class3::{author_dir.name}', items))
        else:
            print('class3 dir not found:', class3_dir)

    results = []
    for label, items in corpora:
        print('Analyzing', label)
        r = analyze_corpus(items, label, outp, sample_size=args.sample_size)
        results.append(r)

    # write summary
    summary_path = outp / 'task1_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # also save CSV table
    rows = []
    for r in results:
        row = {
            'label': r['label'],
            'n_docs': r['n_docs'],
            'words_total': r['words_total'],
            'ttr': r['ttr_sample']['ttr'],
            'hapax': r['ttr_sample']['hapax'],
            'avg_adj_noun_ratio': r['avg_adj_noun_ratio'],
            'avg_dep_tree_depth': r['avg_dep_tree_depth'],
            'flesch_kincaid_grade': r['flesch_kincaid_grade']
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outp / 'metrics_table.csv', index=False)

    # punct heatmap: collect per-corpus per-punct
    punct_df = pd.DataFrame([
        {'label': r['label'], **r['punct_per_1000']} for r in results
    ])
    if not punct_df.empty:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, max(4, len(punct_df)*0.5)))
        sns.heatmap(punct_df.set_index('label')[PUNCT_TYPES], annot=True, fmt='.1f', cmap='magma')
        plt.title('Punctuation per 1000 words')
        plt.tight_layout()
        plt.savefig(outp / 'punct_heatmap.png')
        print('Saved punctuation heatmap to', outp / 'punct_heatmap.png')

    print('Summary written to', summary_path)

if __name__ == '__main__':
    main()
