# Task0 — Presentation Outline (10–12 slides)

## 1) Title / Context
- Task0: The Library of Babel (Authorship + AI-generation analysis)
- Dataset sources: Project Gutenberg texts (Jane Austen, Mark Twain)
- Goal: Build dataset, generate AI text variants, and evaluate detection/"human-likeness"

## 2) What I did vs didn’t (submission checklist)
- Completed: cleaning → topics → prompts → fingerprints → detector tiers (B/C) → Task 4 loop + Personal Test
- Not completed: live Gemini calls (secrets), copy-detection

## 3) Data pipeline (Task 0)
- Clean raw books → normalized paragraphs
- Outputs: `data/cleaned/`, `data/paragraphs.csv`

## 4) Topic extraction (Class 2 creation)
- TF–IDF (uni+bi-grams) → NMF topics
- Manual topic selection from `data/topics/topics_n{N}.json`

## 5) Prompt scaffolding (Class 2 & Class 3)
- JSONL prompts for batch generation
- Class 2: topic-neutral
- Class 3: author-mimic with few-shot examples

## 6) Fingerprints (Task 1)
- Lexical: TTR, hapax
- Readability: FK etc.
- Syntactic: POS ratios, dependency depth
- Surface: punctuation density

## 7) Statistical methodology (Task 1)
- Bootstrap distributions
- Mann–Whitney U tests
- Effect size (Cliff’s delta)
- Artifacts: `data/analysis/*`

## 8) Detector tiers (Tasks 2–3)
- Lightweight TF–IDF + Logistic Regression
- 3-class: `ai_mimic`, `ai_neutral`, `human`
- Binary reporting: $P(ai)=1-P(human)$

## 9) Task 4: Super‑Imposter / convergence
- Gen0…Gen7 iterative rewrite + scoring
- Observed diminishing returns (plateau) after early generations

## 10) Personal Test (SOP)
- Baseline: $P(human)\approx 7.73\%$
- Rewrite: $P(human)=24.08\%$
- Improvement: +16.35 percentage points

## 11) Key takeaways
- Detector score can be shifted by writing style changes
- Scores are model-dependent → best used for draft-to-draft comparison

## 12) Limitations & next steps
- Secure Gemini generation script
- Copy-detection to prevent memorization
- Export Tier A inference bundle
- Multivariate classifier over fingerprint features
