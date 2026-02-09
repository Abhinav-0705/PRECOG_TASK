"""
extract_topics.py

Extract 5-10 topics from cleaned book text using TF-IDF + NMF.
Saves topic keywords to JSON files under data/topics/.

Usage:
    python scripts/extract_topics.py --clean-dir data/cleaned --out-dir data/topics --n-topics 8
"""
import argparse
from pathlib import Path
import json
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def load_texts_and_authors(clean_dir: Path):
    """Return list of documents and metadata, and a mapping author->combined text."""
    docs = []
    metadata = []
    author_texts = defaultdict(list)
    for author_dir in sorted(clean_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        for txt in sorted(author_dir.glob('*.txt')):
            content = txt.read_text(encoding='utf-8', errors='ignore')
            docs.append(content)
            metadata.append({'author': author_dir.name, 'filename': txt.name})
            author_texts[author_dir.name].append(content)
    # join per-author
    for a in list(author_texts.keys()):
        author_texts[a] = '\n'.join(author_texts[a])
    return docs, metadata, author_texts


def simple_tokenize(text: str):
    # lowercase, keep alphabetic tokens length >= 3
    text = text.lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    return tokens


def find_author_specific_tokens(author_texts, min_count=10, dominance=0.8):
    # compute token counts per author and mark tokens that are heavily skewed to one author
    total_counts = Counter()
    per_author = {}
    for author, text in author_texts.items():
        toks = simple_tokenize(text)
        c = Counter(toks)
        per_author[author] = c
        total_counts.update(c)

    author_specific = set()
    for token, total in total_counts.items():
        if total < min_count:
            continue
        for author, c in per_author.items():
            if c[token] / total >= dominance:
                author_specific.add(token)
                break
    return author_specific


def extract_topics(texts, author_texts, n_topics=8, n_words=10):
    # Build custom stopwords: scikit-learn english + common speech tokens + author-specific tokens
    common_extra = {'said', 'wouldn', 'couldn', 'don', 'ain', 'll', 're', 've', 'dont', 'cannot'}
    author_specific = find_author_specific_tokens(author_texts, min_count=8, dominance=0.85)
    stop_words = set()
    # sklearn provides a built-in english stop list if you pass 'english' to TfidfVectorizer,
    # but we will build a final stop set and also pass 'english' to the vectorizer to be safe.
    stop_words.update(common_extra)
    stop_words.update(author_specific)

    # Vectorize: use unigrams + bigrams, filter short tokens, remove extremely common tokens
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b[a-zA-Z]{3,}\\b",
                                 stop_words='english',
                                 max_df=0.85,
                                 min_df=2,
                                 ngram_range=(1, 2))

    X = vectorizer.fit_transform(texts)

    # Remove author-specific tokens from the feature names by zeroing their columns if present
    feature_names = vectorizer.get_feature_names_out()
    cols_to_zero = [i for i, f in enumerate(feature_names) if f in stop_words]
    if cols_to_zero:
        import numpy as np
        X = X.tocsc()
        X.data[np.isin(X.indices, cols_to_zero, assume_unique=False)] = 0
        X = X.tocsr()

    nmf = NMF(n_components=n_topics, random_state=0, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[::-1][:n_words]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics


def main(clean_dir: str, out_dir: str, n_topics: int):
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts, metadata, author_texts = load_texts_and_authors(clean_dir)
    if not texts:
        print('No cleaned texts found in', clean_dir)
        return
    topics = extract_topics(texts, author_texts, n_topics=n_topics)
    # Save a combined topics file
    out_path = out_dir / f'topics_n{n_topics}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'n_topics': n_topics, 'topics': topics, 'metadata': metadata}, f, indent=2)
    print(f'Wrote topics to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', default='data/cleaned')
    parser.add_argument('--out-dir', default='data/topics')
    parser.add_argument('--n-topics', type=int, default=8, help='Number of topics to extract (5-10 recommended)')
    args = parser.parse_args()
    main(args.clean_dir, args.out_dir, args.n_topics)
