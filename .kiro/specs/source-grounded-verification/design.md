# Design Document

## Overview

Source-grounded verification adds a second credibility signal to Veritas Desk alongside the existing TF-IDF + Linear SVM classifier. When a claim is submitted, a new **Retriever** component fetches live articles from curated trusted sources via Google News RSS, a **Similarity Engine** scores how closely each retrieved article agrees with the claim, and a **Fusion Engine** blends the resulting Source_Signal with the existing ML_Signal to produce the final Credibility_Score. Every result now carries an Evidence_Panel listing the retrieved articles so users can judge the evidence directly.

The design is additive: all existing API response fields, ML pipeline code, and frontend behaviour are preserved unchanged. The new logic lives in two new backend modules (`retriever.py`, `similarity.py`) and one new evaluation script (`evaluate_grounded.py`). The Flask endpoint `app.py` and the vanilla-JS frontend `app.js` receive targeted additions only.

---

## Architecture

```mermaid
flowchart TD
    subgraph Client
        UI[Browser — app.js]
    end

    subgraph Flask["Flask (app.py)"]
        EP["/api/analyze POST"]
        ML[ML Pipeline\nTF-IDF + SVM]
        LING[linguistic_markers\nexplain_words]
        FUSION[Fusion Engine]
    end

    subgraph New["New Modules"]
        RET[retriever.py\nRetriever + LRU Cache]
        SIM[similarity.py\nSimilarity Engine]
    end

    GNEWS[(Google News RSS)]
    SOURCES[(sources.json)]

    UI -->|POST {text, url}| EP
    EP --> ML
    EP --> LING
    EP --> RET
    RET -->|query| GNEWS
    RET -->|domain lookup| SOURCES
    RET -->|articles| SIM
    SIM -->|scores + source_signal + confidence_flag| FUSION
    ML -->|p_real| FUSION
    LING -->|ling_score| FUSION
    FUSION -->|credibility, verdict| EP
    EP -->|JSON response| UI
```

The existing synchronous request path is preserved. The RSS fetch happens in the same thread as the ML scoring; the 5-second timeout in `retriever.py` bounds the worst-case latency. If retrieval times out or fails, the system degrades gracefully to ML-only scoring.

---

## Components and Interfaces

### `backend/retriever.py`

Responsibility: query Google News RSS, filter to Trusted_Domains, cache results.

```python
# Public interface
def fetch(claim_text: str) -> list[dict]:
    """Return up to 8 trusted articles for the claim.
    
    Each dict: {title: str, link: str, source_domain: str, reliability: float}
    Returns [] on timeout or any exception.
    Never raises.
    """
```

**Internal design:**
- `_cache_key(claim_text)` — extracts the first 10 non-stopword tokens (using `ENGLISH_STOP_WORDS` from scikit-learn) and returns a `tuple[str, ...]` suitable as a dict key.
- `_fetch_uncached(key_tuple)` — decorated with `@functools.lru_cache(maxsize=256)` via a wrapper; performs the actual RSS request and filtering.
- Trusted_Domain check: `reliability >= 0.80` against the globally loaded `SOURCES` dict.
- Domain extraction reuses the existing `domain_of()` logic from `app.py` (extracted to a shared helper or duplicated; duplication preferred to avoid circular import).
- Max 8 results enforced by slicing after filtering.
- Timeout: `requests.get(..., timeout=5)`.
- All exceptions caught; `logging.exception()` called; empty list returned.

**Relationship to `related_articles()` in `app.py`:**  
The existing `related_articles()` function becomes a one-line delegate:
```python
def related_articles(text: str, n: int = 5) -> list[dict]:
    return [{"title": a["title"], "link": a["link"], "source": a["source_domain"]}
            for a in retriever.fetch(text)[:n]]
```
The function signature and return schema are unchanged so no other call sites break.

---

### `backend/similarity.py`

Responsibility: compute Agreement_Scores and Source_Signal.

```python
# Public interface
def compute_scores(claim_text: str, articles: list[dict]) -> list[float]:
    """Return Agreement_Score for each article, in input order.
    
    Each score in [0.0, 1.0].
    Returns [] when articles is empty.
    """

def compute_source_signal(articles: list[dict], scores: list[float]) -> tuple[float, bool]:
    """Return (source_signal, confidence_flag).
    
    source_signal: reliability-weighted mean of top-3 scores, in [0, 1].
                   0.5 (neutral) when articles is empty.
    confidence_flag: True iff len(articles) >= 2.
    """
```

**Lightweight mode (default):**
1. Apply `clean()` from `preprocess.py` to both the claim and each article title.
2. Build corpus = `[cleaned_claim] + [cleaned_title_i for each article]`.
3. Fit `TfidfVectorizer(min_df=1, sublinear_tf=True)` on the corpus.
4. Extract claim vector (index 0) and article vectors (indices 1…n).
5. Cosine similarity = dot product of L2-normalised vectors (use `sklearn.metrics.pairwise.cosine_similarity`).
6. Clamp result to `[0.0, 1.0]` to absorb floating-point rounding.

**Semantic mode (opt-in):**
- Activated when `SIMILARITY_MODE=semantic` and `sentence_transformers` is importable.
- Uses `SentenceTransformer("all-MiniLM-L6-v2").encode()` on claim and article titles.
- Same cosine similarity computation and clamping.
- If import fails, logs `WARNING: sentence-transformers not available; falling back to lightweight` and uses lightweight mode.

**Source signal computation (`compute_source_signal`):**
1. Pair each score with its article's `reliability` value.
2. Sort pairs by score descending; take top 3 (or all if fewer than 3).
3. `source_signal = sum(score_i * reliability_i) / sum(reliability_i)`.
4. `confidence_flag = len(articles) >= 2`.
5. If `articles` is empty: return `(0.5, False)`.

**Startup logging:**
```python
import logging, os
_MODE = os.environ.get("SIMILARITY_MODE", "lightweight").lower()
logging.getLogger(__name__).info("Similarity mode: %s", _MODE)
```

---

### `backend/app.py` — changes

**Imports added:**
```python
import retriever, similarity
```

**`/api/analyze` additions** (after existing ML scoring block):
```python
articles = retriever.fetch(text)
scores   = similarity.compute_scores(text, articles)
source_signal, confidence_flag = similarity.compute_source_signal(articles, scores)
```

**Fusion block replacement:**  
The new logic replaces the current `parts` list construction:
```python
if confidence_flag and ling_score <= 0.70:
    # Fused formula: ML 55%, Source 45%
    cred = 0.55 * p_real_adj + 0.45 * source_signal
else:
    # Existing formula (ling_weight override + optional URL source nudge)
    # ... existing parts-list code unchanged ...
    cred = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
cred = max(0.0, min(1.0, cred))   # clamp
```
Where `p_real_adj` is the ling-weight-blended value computed just before this block:
```python
p_real_adj = ml_weight * p_real + ling_weight * ling_real
```

**New response fields** (added to `jsonify(...)` call):
```python
source_evidence=[
    {**{k: a[k] for k in ("title","link","source_domain","reliability")},
     "agreement": round(s, 2)}
    for a, s in zip(articles, scores)
],
confidence_flag=confidence_flag,
source_signal=round(source_signal, 3),
```

All existing fields (`verdict`, `credibility`, `p_real`, `model`, `latency_ms`, `markers`, `explanation`, `source`, `related`, `factchecks`) are retained verbatim.

---

### `frontend/app.js` — changes

A new `renderEvidencePanel(d)` helper builds the Evidence_Panel DOM node and is inserted into `render(d)` after the word-explanation block and before the existing `grid` with language flags.

```javascript
function renderEvidencePanel(d) {
  const panel = el("div", "panel");
  panel.append(el("h3", null, "Live source coverage"));

  if (!d.confidence_flag) {
    panel.append(el("p", "err", "Low source coverage — treat the score with caution."));
  }

  if (d.source_evidence && d.source_evidence.length > 0) {
    const ul = el("ul", "plain");
    d.source_evidence.forEach(a => {
      const li   = el("li");
      const link = el("a", null, `${a.title} (${a.source_domain}) — ${Math.round(a.agreement * 100)}%`);
      link.href   = a.link;
      link.target = "_blank";
      link.rel    = "noopener";
      li.append(link);
      ul.append(li);
    });
    panel.append(ul);
  } else {
    panel.append(el("p", "muted", "No live coverage found — verdict based on ML classifier only."));
  }
  return panel;
}
```

Insertion point in `render(d)` — after the `legend` paragraph and before `const grid = el("div", "grid")`:
```javascript
box.append(art, el("p", "legend", "Red words pushed toward fake, green toward real."));
box.append(renderEvidencePanel(d));   // ← new line
const grid = el("div", "grid");       // existing line
```

No other changes to `render()`. The `d.source_evidence`, `d.confidence_flag`, and `d.source_signal` fields are consumed only by the new panel; no existing DOM-building code touches them.

---

### `backend/evaluate_grounded.py`

Offline CLI evaluation script. Accepts a CSV (`text`, `label`, optional `url`) and reports accuracy, precision, recall, F1 for three configurations.

```
python backend/evaluate_grounded.py data/ifnd.csv --sample 500
```

**Key design points:**
- Loads the same `BUNDLE`, `SOURCES`, `retriever`, and `similarity` modules as the live app.
- For each row: runs `retriever.fetch(text)` → `similarity.compute_scores` → `similarity.compute_source_signal` → fusion (same logic as `app.py`).
- If retrieval returns empty (network error, rate-limit): records fallback count, uses ML-only score for that row.
- Computes metrics for three label sets: ML-only predicted labels, Source-only predicted labels (threshold applied to `source_signal`), and Fused labels.
- Writes `models/grounded_metrics.json` in the same schema as `models/metrics.json`.
- `--sample N` uses `random.sample(rows, N)` with a fixed seed for reproducibility.

---

## Data Models

### Article dict (produced by `retriever.fetch`, consumed by `similarity.py` and API response)

```python
{
    "title":         str,   # Article headline from RSS feed
    "link":          str,   # Canonical URL of the article
    "source_domain": str,   # e.g. "thehindu.com"
    "reliability":   float  # In [0, 1]; sourced from sources.json
}
```

### Evidence item (in API response `source_evidence` array)

```python
{
    "title":         str,   # Same as article["title"]
    "link":          str,   # Same as article["link"]
    "source_domain": str,   # Same as article["source_domain"]
    "reliability":   float, # Same as article["reliability"]
    "agreement":     float  # Agreement_Score rounded to 2 dp; in [0.0, 1.0]
}
```

### `/api/analyze` response — new fields

| Field | Type | Description |
|---|---|---|
| `source_evidence` | `list[EvidenceItem]` | Retrieved articles with agreement scores; `[]` when empty |
| `confidence_flag` | `bool` | `true` iff ≥ 2 trusted articles retrieved |
| `source_signal` | `float` | Reliability-weighted mean score, rounded to 3 dp; `0.5` when no articles |

### `models/grounded_metrics.json` schema (output of `evaluate_grounded.py`)

```json
{
  "dataset": "ifnd",
  "rows": 2000,
  "sample": 500,
  "fallback_count": 12,
  "results": [
    {"config": "ml_only",     "accuracy": 0.94, "precision": 0.93, "recall": 0.95, "f1": 0.94},
    {"config": "source_only", "accuracy": 0.71, "precision": 0.70, "recall": 0.73, "f1": 0.71},
    {"config": "fused",       "accuracy": 0.95, "precision": 0.95, "recall": 0.96, "f1": 0.95}
  ]
}
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Retriever domain filter

*For any* RSS feed response containing articles from a mix of trusted and untrusted domains, all articles returned by `retriever.fetch()` shall have a `source_domain` whose `reliability` value in `sources.json` is at least 0.80.

**Validates: Requirements 1.2**

---

### Property 2: Retriever max-8 invariant

*For any* RSS feed response of any length, the list returned by `retriever.fetch()` shall contain at most 8 articles.

**Validates: Requirements 1.3**

---

### Property 3: Score count equals article count

*For any* non-empty list of articles, the list returned by `similarity.compute_scores()` shall have the same length as the input article list.

**Validates: Requirements 2.1**

---

### Property 4: Agreement scores are bounded

*For any* claim text and any non-empty list of articles, every value in the list returned by `similarity.compute_scores()` shall be in the closed interval [0.0, 1.0].

**Validates: Requirements 2.3**

---

### Property 5: Similarity determinism

*For any* claim text of at least 5 words and any non-empty list of articles, calling `similarity.compute_scores()` twice with identical inputs in Lightweight mode shall return identical lists of Agreement_Scores.

**Validates: Requirements 2.7, 9.1**

---

### Property 6: Self-similarity identity

*For any* claim text of at least 5 words, computing `similarity.compute_scores(claim, [article])` where `article["title"]` equals `claim` shall return `[1.0]`.

**Validates: Requirements 9.2**

---

### Property 7: Source signal weighted formula

*For any* list of articles (length ≥ 1) with known `reliability` values and a corresponding list of `Agreement_Scores`, the `source_signal` returned by `similarity.compute_source_signal()` shall equal the reliability-weighted mean of the top-3 scores (or all scores when fewer than 3), computed as `sum(score_i * reliability_i) / sum(reliability_i)`.

**Validates: Requirements 3.1, 3.2**

---

### Property 8: Confidence flag threshold

*For any* articles list, the `confidence_flag` returned by `similarity.compute_source_signal()` shall be `True` if and only if the number of articles is at least 2.

**Validates: Requirements 3.4, 3.5**

---

### Property 9: Fusion formula correctness

*For any* `ml_signal` in [0, 1] and `source_signal` in [0, 1], when `confidence_flag` is `True` and the linguistic marker score is at most 0.70, the `credibility` value in the API response shall equal `round((0.55 * ml_signal + 0.45 * source_signal) * 100)`.

**Validates: Requirements 4.1**

---

### Property 10: Credibility score is clamped

*For any* combination of `ml_signal`, `source_signal`, and `ling_score` inputs, the `credibility` field in the API response shall be an integer in [0, 100].

**Validates: Requirements 4.3**

---

### Property 11: Verdict threshold mapping

*For any* `credibility` value in [0, 100], the `verdict` field shall be `"Likely real"` when `credibility >= 60`, `"Likely fake"` when `credibility < 40`, and `"Unverified"` when `40 <= credibility < 60`.

**Validates: Requirements 4.4**

---

### Property 12: API response backward compatibility

*For any* valid POST body submitted to `/api/analyze`, the JSON response shall contain all of the following fields: `verdict`, `credibility`, `p_real`, `model`, `latency_ms`, `markers`, `explanation`, `source`, `related`, `factchecks`.

**Validates: Requirements 5.6**

---

### Property 13: Evidence item schema completeness

*For any* non-empty `source_evidence` array in the API response, every element shall contain exactly the keys `title`, `link`, `source_domain`, `reliability`, and `agreement`, where `reliability` and `agreement` are both floats in [0, 1] and `agreement` is rounded to 2 decimal places.

**Validates: Requirements 5.2**

---

### Property 14: Evidence links rendered correctly

*For any* non-empty `source_evidence` array passed to `renderEvidencePanel()`, every rendered `<a>` element shall have `target="_blank"`, `rel="noopener"`, and an `href` matching the article's `link` field.

**Validates: Requirements 6.1, 6.4**

---

### Property 15: Score order preservation

*For any* list of articles, the i-th value in the list returned by `similarity.compute_scores()` shall correspond to the i-th article in the input list.

**Validates: Requirements 9.4**

---

## Error Handling

| Failure scenario | Behaviour |
|---|---|
| RSS fetch timeout (> 5 s) | `retriever.fetch()` catches `requests.Timeout`, logs it, returns `[]`. Fusion falls back to ML-only. |
| Any other exception in `retriever.fetch()` | Same: catch-all, `logging.exception()`, return `[]`. Never raises to caller. |
| `source_evidence` empty | `confidence_flag = False`. Fusion uses existing ML + ling formula. `source_evidence: []` in response. |
| `sentence-transformers` not installed when `SIMILARITY_MODE=semantic` | Log `WARNING`, fall back to lightweight. No crash. |
| Credibility computed outside [0, 100] | Clamped by `max(0, min(100, ...))` in fusion block. |
| Malformed RSS feed entry (missing `title` or `source`) | Skip that entry in the filter loop; log at `DEBUG` level. |
| Total analysis > 10 s | The 5 s retrieval timeout caps RSS time. If processing still exceeds budget, the timeout already produced `[]`; response is returned with ML-only score and empty evidence. No separate 10 s deadline required in `app.py` because the RSS timeout is the bottleneck. |
| `evaluate_grounded.py` row retrieval fails | Increment `fallback_count`; score row using ML-only. Record fallback rate in output JSON. |

---

## Testing Strategy

### Unit tests (`backend/tests/test_retriever.py`, `backend/tests/test_similarity.py`, `backend/tests/test_fusion.py`)

Unit tests cover specific scenarios, edge cases, and integration points:

- `retriever.fetch()` with a mock RSS feed containing mixed domains → only trusted domains returned
- `retriever.fetch()` with a mock that raises `requests.Timeout` → empty list, no exception
- `retriever.fetch()` called twice with the same claim → mock called exactly once (cache hit)
- `similarity.compute_scores()` with an empty articles list → `[]`
- `similarity.compute_source_signal()` with empty articles → `(0.5, False)`
- Fusion: `confidence_flag=True, ling_score=0.75` → existing formula used (marker override)
- Fusion: `confidence_flag=False` → `source_signal` has no effect on `credibility`
- API response contains `source_evidence`, `confidence_flag`, `source_signal` fields
- DOM: `renderEvidencePanel` with `source_evidence=[]` → fallback message rendered
- DOM: `renderEvidencePanel` with `confidence_flag=false` → warning notice rendered

### Property-based tests (`backend/tests/test_similarity_properties.py`)

Uses **[Hypothesis](https://hypothesis.readthedocs.io/)** (Python property-based testing library). Minimum 100 examples per property (`@settings(max_examples=100)`).

Each test is tagged with a comment in the format:
`# Feature: source-grounded-verification, Property N: <property_text>`

**Property 1 — Retriever domain filter**
```python
# Feature: source-grounded-verification, Property 1: all fetched articles have reliability >= 0.80
@given(st.lists(st.fixed_dictionaries({
    "title": st.text(min_size=1),
    "link":  st.text(min_size=1),
    "source_domain": st.sampled_from(list(SOURCES.keys()))
}), min_size=0, max_size=20))
def test_retriever_domain_filter(articles):
    # Inject mixed articles into mock RSS; verify only trusted survive
    ...
```

**Property 2 — Max-8 invariant**
```python
# Feature: source-grounded-verification, Property 2: len(fetch_result) <= 8
@given(st.integers(min_value=0, max_value=50))
def test_retriever_max_eight(n_articles):
    ...
```

**Property 3 — Score count equals article count**
```python
# Feature: source-grounded-verification, Property 3: len(scores) == len(articles)
@given(claim=st.text(min_size=20), articles=st.lists(article_strategy(), min_size=1, max_size=10))
def test_score_count(claim, articles):
    assert len(similarity.compute_scores(claim, articles)) == len(articles)
```

**Property 4 — Agreement scores bounded**
```python
# Feature: source-grounded-verification, Property 4: all scores in [0, 1]
@given(claim=st.text(min_size=20), articles=st.lists(article_strategy(), min_size=1, max_size=10))
def test_scores_bounded(claim, articles):
    assert all(0.0 <= s <= 1.0 for s in similarity.compute_scores(claim, articles))
```

**Property 5 — Similarity determinism**
```python
# Feature: source-grounded-verification, Property 5: same inputs → same outputs
@given(claim=valid_claim(), articles=st.lists(article_strategy(), min_size=1, max_size=8))
def test_determinism(claim, articles):
    assert similarity.compute_scores(claim, articles) == similarity.compute_scores(claim, articles)
```

**Property 6 — Self-similarity identity**
```python
# Feature: source-grounded-verification, Property 6: claim compared to itself → score 1.0
@given(claim=valid_claim())
def test_self_similarity(claim):
    article = {"title": claim, "link": "https://example.com", "source_domain": "reuters.com", "reliability": 0.95}
    scores = similarity.compute_scores(claim, [article])
    assert scores == [1.0]
```

**Property 7 — Source signal formula**
```python
# Feature: source-grounded-verification, Property 7: source_signal = weighted mean of top-3
@given(articles=st.lists(article_strategy(), min_size=1, max_size=8),
       scores=st.lists(st.floats(0, 1), min_size=1, max_size=8))
def test_source_signal_formula(articles, scores):
    assume(len(articles) == len(scores))
    signal, _ = similarity.compute_source_signal(articles, scores)
    top3 = sorted(zip(scores, [a["reliability"] for a in articles]), reverse=True)[:3]
    expected = sum(s * r for s, r in top3) / sum(r for _, r in top3)
    assert abs(signal - expected) < 1e-9
```

**Property 8 — Confidence flag threshold**
```python
# Feature: source-grounded-verification, Property 8: flag True iff len(articles) >= 2
@given(articles=st.lists(article_strategy(), min_size=0, max_size=10),
       scores=st.lists(st.floats(0, 1), min_size=0, max_size=10))
def test_confidence_flag(articles, scores):
    assume(len(articles) == len(scores))
    _, flag = similarity.compute_source_signal(articles, scores)
    assert flag == (len(articles) >= 2)
```

**Property 9 — Fusion formula correctness**
```python
# Feature: source-grounded-verification, Property 9: fused credibility = round((0.55*ml+0.45*src)*100)
@given(ml=st.floats(0, 1), src=st.floats(0, 1), ling=st.floats(0, 0.70))
def test_fusion_formula(ml, src, ling):
    result = compute_credibility(ml, src, ling_score=ling, confidence_flag=True)
    assert result == round((0.55 * ml + 0.45 * src) * 100)
```

**Property 10 — Credibility clamped**
```python
# Feature: source-grounded-verification, Property 10: credibility in [0, 100]
@given(ml=st.floats(0, 1), src=st.floats(0, 1), ling=st.floats(0, 1),
       flag=st.booleans())
def test_credibility_clamped(ml, src, ling, flag):
    result = compute_credibility(ml, src, ling, flag)
    assert 0 <= result <= 100
```

**Property 11 — Verdict mapping**
```python
# Feature: source-grounded-verification, Property 11: verdict follows threshold mapping
@given(credibility=st.integers(0, 100))
def test_verdict_mapping(credibility):
    verdict = derive_verdict(credibility)
    if credibility >= 60:
        assert verdict == "Likely real"
    elif credibility < 40:
        assert verdict == "Likely fake"
    else:
        assert verdict == "Unverified"
```

**Property 12 — API backward compatibility**
```python
# Feature: source-grounded-verification, Property 12: all existing fields present
REQUIRED_FIELDS = {"verdict","credibility","p_real","model","latency_ms",
                   "markers","explanation","source","related","factchecks"}
@given(text=valid_claim())
def test_response_fields(text, client):
    resp = client.post("/api/analyze", json={"text": text})
    assert REQUIRED_FIELDS.issubset(resp.json.keys())
```

**Property 13 — Evidence schema completeness**
```python
# Feature: source-grounded-verification, Property 13: each evidence item has required keys
@given(text=valid_claim())
def test_evidence_schema(text, client):
    resp = client.post("/api/analyze", json={"text": text})
    for item in resp.json["source_evidence"]:
        for key in ("title","link","source_domain","reliability","agreement"):
            assert key in item
        assert 0.0 <= item["reliability"] <= 1.0
        assert 0.0 <= item["agreement"] <= 1.0
```

**Property 14 — Evidence links correct**  
*(Frontend — tested with jsdom or similar DOM testing tool)*
```javascript
// Feature: source-grounded-verification, Property 14: all evidence links have target=_blank and rel=noopener
test.prop([fc.array(evidenceItemArb, {minLength: 1})])('evidence links have noopener', (items) => {
  const panel = renderEvidencePanel({ source_evidence: items, confidence_flag: true });
  panel.querySelectorAll('a').forEach(a => {
    expect(a.target).toBe('_blank');
    expect(a.rel).toBe('noopener');
  });
});
```

**Property 15 — Score order preservation**
```python
# Feature: source-grounded-verification, Property 15: score order matches article order
@given(claim=valid_claim(), articles=st.lists(article_strategy(), min_size=2, max_size=8))
def test_order_preservation(claim, articles):
    scores = similarity.compute_scores(claim, articles)
    # Reverse articles and re-score; scores at each position should follow articles
    rev_scores = similarity.compute_scores(claim, list(reversed(articles)))
    assert scores == list(reversed(rev_scores))
```

### Integration tests

- Run `evaluate_grounded.py` on a 50-row sample of `data/ifnd.csv`; verify `models/grounded_metrics.json` is created and contains `results` with three entries.
- POST to `/api/analyze` with a real headline; verify `latency_ms < 6000` and `source_evidence` is a list.
- With `SIMILARITY_MODE=semantic` and `sentence-transformers` absent: verify server starts and falls back to lightweight (check log output).

### Test library

Backend property tests: **Hypothesis** (`pip install hypothesis`). Already compatible with pytest.  
Frontend property tests: **fast-check** (`npm install fast-check --save-dev`), run with jsdom.  
Each property test is configured with `@settings(max_examples=100)` (Hypothesis) or `{numRuns: 100}` (fast-check).
