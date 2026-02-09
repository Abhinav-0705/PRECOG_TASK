# Report — Task0 (Tasks 0–4)

**Workspace:** `Task0/` in `/Users/abhinavchatrathi/Documents/Sem4/PreCog_Task/`  
**Primary artifacts:** `notebooks/Task0_results.ipynb`, `scripts/`, `data/analysis/`

---

## First page — Exactly what I did vs didn’t do (and why)

### What I completed

**Task 0 — Dataset + prompt scaffolding**
- Cleaned raw Gutenberg-style text into paragraph units and created an index CSV.
- Extracted topic keyword lists from the cleaned corpus.
- Built prompt JSONL files for:
	- **Class 2** (topic‑neutral paragraphs)
	- **Class 3** (author‑mimic paragraphs)

**Task 1 — Fingerprint / statistical analysis**
- Implemented a feature (“fingerprint”) pipeline and bootstrap + significance testing to compare distributions across classes.
- Generated analysis artifacts under `data/analysis/` (CSV/JSON summaries).

**Task 2 & Task 3 — Detector tiers (A/B/C)**
- Implemented and trained detector tiers (scikit‑learn based) and integrated them into the results notebook.
- Produced Tier A and Tier ABC evaluation artifacts under `data/analysis/` (datasets, results JSON, confusion matrices, feature importances).
- Saved Tier B and Tier C inference bundles (joblib bundles) for reproducibility.

**Task 4 — Turing Test / Super‑Imposter + Personal Test**
- Ran the iterative generation/scoring loop inside `notebooks/Task0_results.ipynb` (Gen0…Gen7) and documented convergence.
- Implemented a **Personal Test** workflow (score an SOP draft, then re‑score an edited draft).
- Fixed probability reporting to correctly handle the **3‑class** model output and provide a binary view $P(\text{human})$ vs $P(\text{ai})$.

### What I did not complete (intentionally deferred)

**Gemini API generation runs (Class 2 / Class 3)**
- I did **not** execute live Gemini API calls inside this repo because it would require secrets (API key) and can’t be reproduced safely in a public submission.
- Instead, I prepared the prompt JSONL files and a clean “call the API” contract in `README.md`.

**Tier A vs TierABC clarification (and inference use)**
- In this repo, **Tier A** is a *binary* numeric-feature classifier (`human` vs `ai`) trained by `scripts/tierA_trainer.py`.
- `scripts/tierABC_trainer.py` trains a *3‑class* numeric-feature classifier (`human`, `ai_neutral`, `ai_mimic`). It is **not “Tier A only”**; it’s the combined multiclass experiment.
- While Tier A / TierABC results exist as analysis artifacts (`tierA_results.json`, `tierABC_results.json`, etc.), they were not packaged/used as the primary inference pathway in the **Personal Test** section. The Personal Test uses the Tier C joblib bundle for consistent scoring, and Tier B was also tested.

**Automated plagiarism/copy detection**
- I did not finalize a full n‑gram overlap / near‑duplicate detector between generated outputs and source paragraphs. This is a recommended follow‑up for production use.

---

## Methodology

### Data processing (Task 0)

1. **Cleaning (Gutenberg text → paragraphs)**
- Removed boilerplate headers/footers.
- Normalized whitespace.
- Split into paragraph blocks (blank-line separators).
- Outputs:
	- `data/cleaned/{author}/...`
	- `data/paragraphs.csv` (index of paragraph samples)

2. **Topic extraction (topic lists for Class 2)**
- Vectorized paragraphs using TF‑IDF (unigrams + bigrams).
- Extracted topic components using NMF.
- Filtered overly author‑specific tokens where possible.
- Output: `data/topics/topics_n{N}.json`.

3. **Prompt generation (Class 2 / Class 3)**
- Wrote prompts as JSONL lines for easy batching.
- **Class 2:** generate a neutral paragraph about a chosen topic.
- **Class 3:** generate a paragraph mimicking a target author, with a few-shot excerpt.

### Fingerprint / statistical evaluation (Task 1)

Computed per‑sample fingerprints spanning:
- **Lexical:** TTR / hapax counts.
- **Readability:** FK and related readability scores.
- **Syntactic:** POS ratios, dependency depth (via spaCy).
- **Surface:** punctuation density and character‑level markers.

Then:
- Bootstrapped metric distributions per class.
- Compared class pairs with Mann–Whitney U tests.
- Reported effect sizes (Cliff’s delta).

### Detector tiers (Tasks 2–3)

I implemented three tiers conceptually, with the practical emphasis on **Tier B** and **Tier C**:

- **Tier B:** TF‑IDF + Logistic Regression (3‑class: `ai_mimic`, `ai_neutral`, `human`), stored as a joblib bundle for inference.
- **Tier C:** Same modeling family but trained/tuned as the “main” detector used in the final notebook and Personal Test workflow.

**Probability reporting (important fix):**
- The detector is **3‑class**, so the “AI probability” is not a single class.
- For the report, I collapse it into:
	$$P(\text{ai}) = 1 - P(\text{human}) = P(\text{ai\_neutral}) + P(\text{ai\_mimic})$$
- This guarantees $P(\text{ai}) + P(\text{human}) = 1$ and avoids mis-indexing pitfalls.

### Task 4 — Turing Test / Super‑Imposter + Personal Test

1. **Iterative generation/scoring loop**
- Ran a generation loop (Gen0…Gen7) and scored each generation under Tier C.
- Tracked convergence (improvements plateauing across generations).

2. **Personal Test (SOP)**
- Scored a baseline SOP draft.
- Scored a manually rewritten candidate.
- Compared $P(\text{human})$ changes as a measure of “humanization”.

---

## Results & key findings

### Task 0 findings (dataset/prompting)

- Topic extraction is sensitive to corpus imbalance (e.g., over‑representation of a single author can dominate topic terms).
- Filtering proper nouns / author-specific terms improves topic generality but does not fully remove leakage.

### Task 1 findings (fingerprints)

- Multiple fingerprint metrics show measurable distribution shifts across classes.
- Bootstrap CIs and non‑parametric tests help separate “real signal” from sampling variance.

### Tasks 2–3 findings (detectors)

- A lightweight TF‑IDF + Logistic Regression model is sufficient to obtain meaningful separation between human vs AI-generated classes.
- In practice, interpretation should be **relative** (used to compare drafts) rather than asserting absolute truth.

### Task 4 findings (Super‑Imposter + Personal Test)

#### Convergence observation
- Across successive generations, the best-scoring candidates improved early and then plateaued, indicating diminishing returns from repeated model-driven rewrites.

#### Personal Test (documented numbers)

Using Tier C with binary reporting ($P(\text{ai}) = 1 - P(\text{human})$):

- **Baseline SOP (previous run):**
	- $P(\text{human}) \approx 7.73\%$  
	- $P(\text{ai}) \approx 92.27\%$  
	- Predicted: `ai`

- **Manually rewritten candidate (latest run):**
	- Candidate length: 3,890 chars
	- $P(\text{human}) = 24.08\%$  
	- $P(\text{ai}) = 75.92\%$  
	- Predicted: `ai`

- **Change:**
	- $\Delta P(\text{human}) \approx +16.35$ percentage points

**Interpretation:** manual edits can significantly increase “human-likeness” under the detector, but the draft still retains enough detectable signals to remain AI‑leaning.

---

## Reproducibility (how to run)

The repo provides scripts and notebooks. Typical flow:

1. Create and activate a venv, install deps (see `README.md`).
2. Run:
	 - `scripts/clean_text.py` → produces cleaned paragraphs
	 - `scripts/extract_topics.py` → produces `data/topics/*.json`
	 - `scripts/prepare_gemini_prompts.py` → produces prompt JSONL
3. Run training scripts (Tier B/C trainers) as needed.
4. Open `notebooks/Task0_results.ipynb` to view the full end-to-end outputs.

---

## Limitations & next steps

- **Secure generation:** add an opt‑in `scripts/call_gemini.py` that reads API key from env vars and writes `data/generated/` with full provenance.
- **Copy detection:** implement n‑gram overlap / MinHash to flag memorization or near‑copying from the source corpus.
- **Tier A packaging:** export Tier A as a stable inference bundle to make cross-tier comparisons consistent.
- **Multivariate evaluation:** train a classifier on fingerprint features to quantify separability with cross‑validation.

---


