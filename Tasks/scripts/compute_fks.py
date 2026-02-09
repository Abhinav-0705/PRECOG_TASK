#!/usr/bin/env python3
# compute_fks.py - compute ASL/ASW/FK per label by reading corpora (fallback when textstat missing)
import re, json
from pathlib import Path
import csv

ROOT = Path('/Users/abhinavchatrathi/Documents/Sem4/PreCog_Task/Task0')
SUMMARY = ROOT / 'data' / 'analysis' / 'task1_summary.json'
OUT = ROOT / 'data' / 'analysis' / 'metrics_table_with_fk.csv'

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
    words = re.findall(r"[A-Za-z']+", text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    if not words or not sentences:
        return None, None, None
    ASL = len(words)/len(sentences)
    syllables = sum(count_syllables(w) for w in words)
    ASW = syllables/len(words)
    FK = 0.39*ASL + 11.8*ASW - 15.59
    return ASL, ASW, FK

# load summary
if not SUMMARY.exists():
    print('Summary not found:', SUMMARY)
    raise SystemExit(1)
with open(SUMMARY,'r',encoding='utf-8') as fh:
    summary = json.load(fh)

rows = []
for rec in summary:
    label = rec.get('label')
    joined = ''
    if label.startswith('class0::'):
        author = label.split('::',1)[1]
        folder = ROOT / 'Class0' / author
        txts = sorted(folder.rglob('*.txt')) if folder.exists() else []
        joined = '\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in txts)
    elif label.startswith('class2::'):
        topic = label.split('::',1)[1]
        p = ROOT / 'Class2' / (topic + '.txt')
        if p.exists():
            joined = p.read_text(encoding='utf-8',errors='ignore')
    elif label.startswith('class3::'):
        author = label.split('::',1)[1]
        folder = ROOT / 'Class3' / author
        txts = sorted(folder.rglob('*.txt')) if folder.exists() else []
        joined = '\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in txts)
    ASL, ASW, FK = fk_grade_from_text(joined)
    rows.append({'label': label, 'ASL': ASL, 'ASW': ASW, 'FK': FK})

# write CSV and print
with open(OUT,'w',encoding='utf-8',newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['label','ASL','ASW','FK'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print('Wrote', OUT)
for r in rows:
    print(r['label'], 'FK=', r['FK'], 'ASL=', r['ASL'], 'ASW=', r['ASW'])
