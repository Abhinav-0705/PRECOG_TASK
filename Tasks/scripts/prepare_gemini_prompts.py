"""
prepare_gemini_prompts.py

Prepare JSONL prompt files for Gemini Class 2 and Class 3 outputs.

- Class 2: Neutral-style paragraphs on chosen topics.
- Class 3: Author-mimic paragraphs on the same topics, include a short few-shot context example from the author's text.

Output files:
  data/prompts/class2.jsonl
  data/prompts/class3.jsonl

Usage:
  python scripts/prepare_gemini_prompts.py --topics-file data/topics/topics_n8.json --paragraphs data/paragraphs.csv --out-dir data/prompts --n-samples 500

Note: This script only prepares prompts and metadata; you must call the Gemini API with these prompts separately.
"""
import argparse
from pathlib import Path
import json
import csv
import random

PROMPT_CLASS2 = (
    "Write a single coherent paragraph (100-200 words) about the following topic."
    " Stick to the topic and avoid quoting other texts.\n\nTopic: {topic}\n\nParagraph:"
)

PROMPT_CLASS3_TEMPLATE = (
    "You are a creative writer who will write a single paragraph (100-200 words) in the style of {author}.\n"
    "Here are two short examples from {author} to set style and tone:\n\nExample 1:\n{ex1}\n\nExample 2:\n{ex2}\n\nNow write a paragraph on the topic below that mimics {author}'s style, voice, and syntactic patterns. Do not copy phrasing verbatim; produce original text.\n\nTopic: {topic}\n\nParagraph:"
)


def sample_examples_for_author(paragraphs_csv, author, n=2):
    # read CSV and sample paragraphs for author
    rows = []
    with open(paragraphs_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['author'] == author:
                rows.append(r['paragraph'])
    if not rows:
        return ["".join([]), "".join([])]
    return random.sample(rows, min(n, len(rows)))


def main(topics_file, paragraphs_csv, out_dir, n_samples):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(topics_file, encoding='utf-8') as f:
        topics_data = json.load(f)
    topics = topics_data.get('topics', [])
    metadata = topics_data.get('metadata', [])

    # Flatten topics into human-readable strings (join keywords)
    topic_texts = [', '.join(t) for t in topics]
    # Create Class2 JSONL: neutral prompts sampling topics round-robin
    class2_path = out_dir / 'class2.jsonl'
    class3_path = out_dir / 'class3.jsonl'

    # Build author list from metadata
    authors = list({m['author'] for m in metadata})

    # For class2: write n_samples prompts
    with open(class2_path, 'w', encoding='utf-8') as f2:
        for i in range(n_samples):
            topic = topic_texts[i % len(topic_texts)]
            prompt = PROMPT_CLASS2.format(topic=topic)
            item = {'id': i+1, 'class': 2, 'topic': topic, 'prompt': prompt}
            f2.write(json.dumps(item, ensure_ascii=False) + '\n')

    # For class3: sample an author for each prompt and attach examples
    with open(class3_path, 'w', encoding='utf-8') as f3:
        for i in range(n_samples):
            topic = topic_texts[i % len(topic_texts)]
            author = authors[i % len(authors)] if authors else 'unknown'
            exs = sample_examples_for_author(paragraphs_csv, author, n=2)
            ex1 = exs[0] if exs else ''
            ex2 = exs[1] if len(exs) > 1 else ex1
            prompt = PROMPT_CLASS3_TEMPLATE.format(author=author, ex1=ex1, ex2=ex2, topic=topic)
            item = {'id': i+1, 'class': 3, 'author': author, 'topic': topic, 'prompt': prompt}
            f3.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Wrote class2 prompts to {class2_path} and class3 prompts to {class3_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics-file', default='data/topics/topics_n8.json')
    parser.add_argument('--paragraphs', default='data/paragraphs.csv')
    parser.add_argument('--out-dir', default='data/prompts')
    parser.add_argument('--n-samples', type=int, default=500)
    args = parser.parse_args()
    main(args.topics_file, args.paragraphs, args.out_dir, args.n_samples)
