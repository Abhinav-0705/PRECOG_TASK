#!/usr/bin/env python3
"""Check notebook environment and data/analysis files.
Prints whether pandas is importable, which DATA_DIR is found, summary of task1_summary.json, head of metrics_table.csv, and whether punct_heatmap.png exists.
"""
import sys
from pathlib import Path
import json
import csv

# Try pandas
try:
    import pandas as pd
    has_pandas = True
except Exception:
    has_pandas = False

candidates = [Path('data') / 'analysis', Path.cwd() / 'data' / 'analysis', Path.cwd().parent / 'data' / 'analysis']
DATA_DIR = next((p for p in candidates if p.exists()), Path('data') / 'analysis')
print('Pandas importable:', has_pandas)
print('Using DATA_DIR =', DATA_DIR)

summary_path = DATA_DIR / 'task1_summary.json'
metrics_path = DATA_DIR / 'metrics_table.csv'
heatmap_path = DATA_DIR / 'punct_heatmap.png'

if summary_path.exists():
    try:
        with open(summary_path, 'r', encoding='utf-8') as fh:
            task1 = json.load(fh)
        print('\nLoaded task1_summary.json with', len(task1), 'records')
        # print a concise table: label and FK
        print('\nLabel - Flesch-Kincaid grade')
        for rec in task1:
            label = rec.get('label')
            fk = rec.get('flesch_kincaid_grade')
            print(f'{label} -> {fk}')
    except Exception as e:
        print('Error reading task1_summary.json:', e)
else:
    print('\ntask1_summary.json not found at', summary_path)

if metrics_path.exists():
    try:
        with open(metrics_path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        print('\nmetrics_table.csv loaded, rows:', len(rows)-1)
        # print header and first 3 data rows
        for r in rows[:4]:
            print(','.join(r))
    except Exception as e:
        print('Error reading metrics_table.csv:', e)
else:
    print('\nmetrics_table.csv not found at', metrics_path)

print('\npunct_heatmap.png exists:', heatmap_path.exists())

# Exit code 0
