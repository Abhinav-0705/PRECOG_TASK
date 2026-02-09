"""
clean_text.py

Strip Project Gutenberg headers/footers, normalize whitespace, split into paragraphs,
and write cleaned paragraphs and a CSV index for downstream use.

Usage:
    python scripts/clean_text.py --data-dir data --out-dir data/cleaned

This script assumes source files are under `data/{author}/...` and will recurse.
"""
import re
import os
import argparse
import csv
from pathlib import Path

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\n", re.I | re.S)
GUTENBERG_END_RE = re.compile(r"\n\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*", re.I | re.S)

def strip_gutenberg_markers(text: str) -> str:
    # Remove common Gutenberg header/footer blocks when present
    m = GUTENBERG_START_RE.search(text)
    if m:
        text = text[m.end():]
    m2 = GUTENBERG_END_RE.search(text)
    if m2:
        text = text[:m2.start()]
    # Fallback heuristics: remove header up to the word "***" blocks
    text = re.sub(r"^.*?\*\*\*", "", text, flags=re.S)
    text = re.sub(r"\*\*\*.*$", "", text, flags=re.S)
    return text


def normalize_whitespace(text: str) -> str:
    # Normalize newlines and replace multiple blank lines with two newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse many newlines to two newlines (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim
    text = text.strip()
    return text


def split_paragraphs(text: str):
    # Use double newlines as paragraph separator
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def process_file(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    text = strip_gutenberg_markers(text)
    text = normalize_whitespace(text)
    paras = split_paragraphs(text)
    return paras


def main(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paragraphs_csv = out_dir.parent / 'paragraphs.csv'

    rows = []
    pid = 0
    for author_dir in sorted(data_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        for root, _, files in os.walk(author_dir):
            for fname in sorted(files):
                if not fname.lower().endswith('.txt'):
                    continue
                fpath = Path(root) / fname
                try:
                    paras = process_file(fpath)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
                    continue
                rel_out_dir = out_dir / author_dir.name
                rel_out_dir.mkdir(parents=True, exist_ok=True)
                out_file = rel_out_dir / (fpath.stem + '_clean.txt')
                out_file.write_text('\n\n'.join(paras), encoding='utf-8')
                for p in paras:
                    pid += 1
                    rows.append({'id': pid, 'author': author_dir.name, 'source': str(fpath.relative_to(data_dir)), 'paragraph': p})

    # Write CSV
    with open(paragraphs_csv, 'w', encoding='utf-8', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['id','author','source','paragraph'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} paragraphs to {paragraphs_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Path to data directory (default: data)')
    parser.add_argument('--out-dir', default='data/cleaned', help='Output directory for cleaned texts')
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
