# Veritas Desk: Fake News Detection using NLP + ML

Academic prototype. Paste a headline or article and get a credibility score,
a verdict, word-level explanations, language red flags, source reliability,
and links to trusted coverage and fact-checkers.

## Pipeline

1. Input: web form (headline/article + optional source URL)
2. Preprocess (`backend/preprocess.py`): lowercase, URL removal, tokenise, stop-words, Porter stemming
3. Features: TF-IDF with unigrams + bigrams
4. Models compared (`backend/train.py`): Logistic Regression, Linear SVM, Random Forest. Duplicates removed, 80:20 static test split, best model chosen on a validation split (never on the test set)
5. Explainability (`backend/explain.py`): LR coefficient x TF-IDF per word, plus clickbait/CAPS/sensational markers
6. Credibility score = 65% model + 20% language cleanliness + 15% source reliability (weights renormalise if no link)
7. Remediation (`backend/app.py`): related articles via Google News RSS, fact-check links, SQLite history and unreliable-source watchlist (3+ fake flags)

## Setup (Windows)

```
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python backend\\train.py        # trains on the toy seed set; see below for real data
python backend\\app.py
```

Open http://127.0.0.1:5000  (macOS/Linux: `source venv/bin/activate`)

## Static training (do this first, as much data as you can)

Put any of these in `data/`, then run `python backend\\train.py`:

* `Fake.csv` + `True.csv` (ISOT) - agency tags like "(Reuters) -" are stripped so the model cannot cheat
* any extra CSV with columns `text,label` (0 = fake, 1 = real), e.g. WELFake or a binarised LIAR
Mixing 2-3 datasets matters more than tuning models: it is what makes the model generalise.
Check `models/metrics.json`: `demo\_only: true` means it trained on the 24-sentence toy set.

## Live learning (after deployment)

* Each result has "Actually real / Actually fake" buttons. Corrections go to `data/live.csv` (weight 3x).
* "Fetch trusted headlines and retrain now" (or `python backend\\learn.py harvest` then `retrain`) adds fresh headlines
from sources rated >= 0.90 as REAL examples, retrains and hot-reloads the model without restarting Flask.
* Rules: the model never trains on its own predictions; live rows go into training only, the test set stays static.
* Limitation to state in your report: RSS only supplies real examples, so fake-side learning depends on human feedback.

## Use a real dataset (needed for meaningful accuracy)

Download ISOT from Kaggle ("Fake and Real News Dataset"), put `Fake.csv` and `True.csv` in `data/`,
then run `python backend\\train.py` again. Or provide `data/news.csv` with columns `text,label` (0 = fake, 1 = real).
The built-in seed set has only 24 sentences: it is a demo, not a result.

## Folder structure

```
backend/   app.py  train.py  learn.py  preprocess.py  explain.py  seed\_data.py  sources.json
frontend/  index.html  style.css  app.js
data/      put datasets here (history.db is created automatically)
models/    model.joblib + metrics.json (created by train.py)
```

## Ideas to extend (for viva / report)

* Swap TF-IDF for BERT embeddings (`sentence-transformers`) and compare latency vs accuracy
* Add the LIAR dataset for 6-class output (True ... Pants-on-Fire)
* Cross-domain test: train on ISOT, test on Indian headlines to show domain imbalance
* Limitations: the model learns writing style, not facts. It cannot verify claims on its own.

"# fake-detection" 
"# fake-detection" 
"# fake-detection" 
"# fake-detection" 
"# fake-detection" 
