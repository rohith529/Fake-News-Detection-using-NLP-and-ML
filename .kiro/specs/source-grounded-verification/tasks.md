# Implementation Plan: Source-Grounded Verification

## Overview

Implement source-grounded verification by adding `retriever.py` and `similarity.py` as standalone modules, wiring them into `app.py`'s `/api/analyze` endpoint, extending the JSON response with evidence fields, rendering an Evidence Panel in `app.js`, and providing an offline evaluation CLI in `evaluate_grounded.py`. All existing behaviour, fields, and passing tests must be preserved throughout.

---

## Tasks

- [x] 1. Create `backend/retriever.py` — Retriever with LRU cache
  - [x] 1.1 Implement `_cache_key()`, `fetch()`, and trusted-domain filtering
    - Create `backend/retriever.py` with the public function `fetch(claim_text: str) -> list[dict]`.
    - `_cache_key(claim_text)` strips stop-words using `ENGLISH_STOP_WORDS` from scikit-learn, takes the first 10 non-stopword tokens, and returns a `tuple[str, ...]`.
    - `fetch()` wraps an `@functools.lru_cache(maxsize=256)` inner function that calls `requests.get("https://news.google.com/rss/search", timeout=5, params=...)` and parses the response with `feedparser`.
    - Trusted-domain filter: keep only entries whose `source` domain has `reliability >= 0.80` in `SOURCES` (loaded once at module level from `sources.json` via `Path(__file__).parent / "sources.json"`).
    - Domain extraction uses a local `_domain_of(url)` helper (same logic as `app.py`'s `domain_of()`) to avoid a circular import.
    - Slice filtered results to at most 8 items before returning.
    - Return dicts with keys `title`, `link`, `source_domain`, `reliability`.
    - Catch all exceptions with `logging.exception()`; return `[]` and never raise to the caller.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

  - [ ]* 1.2 Write unit tests for `retriever.fetch()` in `backend/tests/test_retriever.py`
    - Test: mock RSS feed with mixed trusted/untrusted domains → only trusted domains appear in result.
    - Test: mock raises `requests.Timeout` → returns `[]`, no exception propagated.
    - Test: call `fetch()` twice with the same claim → underlying network mock called exactly once (cache hit).
    - Test: mock returns more than 8 articles → result has length ≤ 8.
    - Test: mock raises a generic `Exception` → returns `[]`, logs via `logging.exception`.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 1.3 Write property test for Retriever domain filter (Property 1)
    - File: `backend/tests/test_similarity_properties.py`
    - `# Feature: source-grounded-verification, Property 1: all fetched articles have reliability >= 0.80`
    - Use `@given` with a list of article dicts whose `source_domain` is sampled from `SOURCES.keys()`. Inject via monkeypatching the RSS response; verify every item in `fetch()` output has `reliability >= 0.80`.
    - `@settings(max_examples=100)`
    - _Requirements: 1.2_

  - [ ]* 1.4 Write property test for Retriever max-8 invariant (Property 2)
    - File: `backend/tests/test_similarity_properties.py`
    - `# Feature: source-grounded-verification, Property 2: len(fetch_result) <= 8`
    - `@given(st.integers(min_value=0, max_value=50))` controlling the number of trusted entries injected.
    - `@settings(max_examples=100)`
    - _Requirements: 1.3_

- [x] 2. Create `backend/similarity.py` — Similarity Engine
  - [x] 2.1 Implement `compute_scores()` in Lightweight mode
    - Create `backend/similarity.py`.
    - Read `SIMILARITY_MODE = os.environ.get("SIMILARITY_MODE", "lightweight").lower()` at module load.
    - Log `logging.getLogger(__name__).info("Similarity mode: %s", _MODE)` at import time.
    - `compute_scores(claim_text: str, articles: list[dict]) -> list[float]`:
      - Return `[]` immediately when `articles` is empty.
      - Apply `clean()` from `preprocess.py` to both `claim_text` and each `article["title"]`.
      - Build corpus = `[cleaned_claim] + [cleaned_title_i for each article]`.
      - Fit `TfidfVectorizer(min_df=1, sublinear_tf=True)` on the corpus.
      - Compute cosine similarity between claim vector (index 0) and each article vector using `sklearn.metrics.pairwise.cosine_similarity`.
      - Clamp each score to `[0.0, 1.0]`.
      - Return scores in input-article order.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7, 9.1, 9.4_

  - [x] 2.2 Implement `compute_source_signal()` and Semantic mode fallback
    - `compute_source_signal(articles: list[dict], scores: list[float]) -> tuple[float, bool]`:
      - If `articles` is empty, return `(0.5, False)`.
      - Pair each score with `article["reliability"]`; sort pairs by score descending; take top 3 (or all if fewer).
      - `source_signal = sum(score_i * reliability_i) / sum(reliability_i)`.
      - `confidence_flag = len(articles) >= 2`.
      - Return `(source_signal, confidence_flag)`.
    - Add Semantic mode branch in `compute_scores()`: when `SIMILARITY_MODE == "semantic"`, attempt `from sentence_transformers import SentenceTransformer` and use `SentenceTransformer("all-MiniLM-L6-v2").encode()`. On any `ImportError` or init exception, log `WARNING: sentence-transformers not available; falling back to lightweight` to `stderr` and proceed with lightweight logic.
    - _Requirements: 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 2.3 Write unit tests for `similarity.py` in `backend/tests/test_similarity.py`
    - Test: `compute_scores()` with empty articles list → `[]`.
    - Test: `compute_source_signal()` with empty articles → `(0.5, False)`.
    - Test: `compute_source_signal()` with 1 article → `confidence_flag == False`.
    - Test: `compute_source_signal()` with 2 articles → `confidence_flag == True`.
    - Test: `compute_source_signal()` with 4 articles with known scores/reliabilities → `source_signal` equals the manually computed reliability-weighted mean of the top-3.
    - Test: when `SIMILARITY_MODE=semantic` and `sentence_transformers` is not installed → module loads without error, `compute_scores()` returns float list (lightweight fallback).
    - _Requirements: 2.6, 3.3, 3.4, 3.5, 7.3_

  - [ ]* 2.4 Write property tests for `similarity.py` (Properties 3–8, 15) in `backend/tests/test_similarity_properties.py`
    - **Property 3 — Score count equals article count**
      - `# Feature: source-grounded-verification, Property 3: len(scores) == len(articles)`
      - `@given(claim=st.text(min_size=20), articles=st.lists(article_strategy(), min_size=1, max_size=10))`
      - Assert `len(similarity.compute_scores(claim, articles)) == len(articles)`.
    - **Property 4 — Agreement scores bounded**
      - `# Feature: source-grounded-verification, Property 4: all scores in [0, 1]`
      - Assert `all(0.0 <= s <= 1.0 for s in scores)` for any non-empty articles list.
    - **Property 5 — Similarity determinism**
      - `# Feature: source-grounded-verification, Property 5: same inputs → same outputs`
      - Call `compute_scores` twice with identical inputs; assert the two lists are equal.
    - **Property 6 — Self-similarity identity**
      - `# Feature: source-grounded-verification, Property 6: claim compared to itself → score 1.0`
      - `@given(claim=valid_claim())` where `valid_claim()` generates text of ≥ 5 words.
      - Build article with `title == claim`; assert `compute_scores(claim, [article]) == [1.0]`.
    - **Property 7 — Source signal formula**
      - `# Feature: source-grounded-verification, Property 7: source_signal = weighted mean of top-3`
      - `@given(articles, scores)` with `assume(len(articles) == len(scores))`; recompute expected from first principles; assert `abs(signal - expected) < 1e-9`.
    - **Property 8 — Confidence flag threshold**
      - `# Feature: source-grounded-verification, Property 8: flag True iff len(articles) >= 2`
      - Assert `flag == (len(articles) >= 2)` for all article counts 0–10.
    - **Property 15 — Score order preservation**
      - `# Feature: source-grounded-verification, Property 15: score order matches article order`
      - `@given(claim=valid_claim(), articles=st.lists(article_strategy(), min_size=2, max_size=8))`
      - Reverse the article list and re-score; assert original scores equal the reversed re-score list reversed.
    - All tests use `@settings(max_examples=100)`.
    - _Requirements: 2.1, 2.3, 2.7, 3.1, 3.2, 3.4, 3.5, 9.1, 9.2, 9.4_

- [x] 3. Checkpoint — Verify retriever and similarity modules in isolation
  - Ensure all tests in `backend/tests/test_retriever.py`, `backend/tests/test_similarity.py`, and `backend/tests/test_similarity_properties.py` pass. Ask the user if any unexpected failures arise before proceeding.

- [x] 4. Modify `backend/app.py` — integrate retriever and similarity, update fusion, extend response
  - [x] 4.1 Add imports and wire retriever + similarity into `/api/analyze`
    - Add `import retriever, similarity` near the top of `app.py` (after existing imports).
    - In `/api/analyze`, after the existing ML scoring block, insert:
      ```python
      articles = retriever.fetch(text)
      scores   = similarity.compute_scores(text, articles)
      source_signal, confidence_flag = similarity.compute_source_signal(articles, scores)
      ```
    - _Requirements: 1.1, 2.1, 3.1, 5.1_

  - [x] 4.2 Replace the fusion block with the new weighted formula
    - Compute `p_real_adj = ml_weight * p_real + ling_weight * ling_real` before the branch (this value is what the existing `parts` list produced as the weighted combination of ML and ling signals).
    - Replace the `parts`-based credibility computation with:
      ```python
      if confidence_flag and ling_score <= 0.70:
          cred = 0.55 * p_real_adj + 0.45 * source_signal
      else:
          # existing parts-list logic (including optional URL source nudge) unchanged
          ...
      cred = max(0.0, min(1.0, cred))
      ```
    - The `verdict` derivation and all subsequent DB writes remain unchanged.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 4.3 Extend the `jsonify(...)` response with new fields
    - Add to the existing `jsonify(...)` call:
      ```python
      source_evidence=[
          {**{k: a[k] for k in ("title", "link", "source_domain", "reliability")},
           "agreement": round(s, 2)}
          for a, s in zip(articles, scores)
      ],
      confidence_flag=confidence_flag,
      source_signal=round(source_signal, 3),
      ```
    - Preserve all existing fields: `verdict`, `credibility`, `p_real`, `model`, `latency_ms`, `markers`, `explanation`, `source`, `related`, `factchecks`.
    - Delegate `related_articles()` to `retriever.fetch()`: replace the body of the existing `related_articles()` with `return [{"title": a["title"], "link": a["link"], "source": a["source_domain"]} for a in retriever.fetch(text)[:n]]`. Function signature unchanged.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 4.4 Write unit tests for the modified `app.py` in `backend/tests/test_fusion.py`
    - Test: `confidence_flag=True, ling_score=0.40` → `credibility` equals `round((0.55 * p_real_adj + 0.45 * source_signal) * 100)`.
    - Test: `confidence_flag=True, ling_score=0.75` → existing formula used (marker override), `source_signal` has no effect.
    - Test: `confidence_flag=False` → `source_signal` has no effect on `credibility`.
    - Test: POST to `/api/analyze` with a 10-word claim → response JSON contains `source_evidence` (list), `confidence_flag` (bool), `source_signal` (float).
    - Test: all existing fields (`verdict`, `credibility`, `p_real`, `model`, `latency_ms`, `markers`, `explanation`, `source`, `related`, `factchecks`) present in response.
    - _Requirements: 4.1, 4.2, 4.5, 5.1, 5.5, 5.6, 5.7_

  - [ ]* 4.5 Write property tests for fusion and response in `backend/tests/test_similarity_properties.py`
    - **Property 9 — Fusion formula correctness**
      - `# Feature: source-grounded-verification, Property 9: fused credibility = round((0.55*ml+0.45*src)*100)`
      - Extract a pure `compute_credibility(ml, src, ling_score, confidence_flag)` helper from `app.py` for testability (or test via Flask test client with mocked `retriever`/`similarity`).
      - `@given(ml=st.floats(0,1), src=st.floats(0,1), ling=st.floats(0,0.70))` with `confidence_flag=True`.
      - Assert `result == round((0.55 * ml + 0.45 * src) * 100)`.
    - **Property 10 — Credibility clamped**
      - `# Feature: source-grounded-verification, Property 10: credibility in [0, 100]`
      - `@given(ml=st.floats(0,1), src=st.floats(0,1), ling=st.floats(0,1), flag=st.booleans())`
      - Assert `0 <= result <= 100`.
    - **Property 11 — Verdict threshold mapping**
      - `# Feature: source-grounded-verification, Property 11: verdict follows threshold mapping`
      - `@given(credibility=st.integers(0,100))`
      - Assert verdict is `"Likely real"` when `credibility >= 60`, `"Likely fake"` when `credibility < 40`, `"Unverified"` when `40 <= credibility < 60`.
    - **Property 12 — API backward compatibility**
      - `# Feature: source-grounded-verification, Property 12: all existing fields present`
      - `REQUIRED_FIELDS = {"verdict","credibility","p_real","model","latency_ms","markers","explanation","source","related","factchecks"}`
      - `@given(text=valid_claim())` using Flask test client.
      - Assert `REQUIRED_FIELDS.issubset(resp.json.keys())`.
    - **Property 13 — Evidence schema completeness**
      - `# Feature: source-grounded-verification, Property 13: each evidence item has required keys`
      - `@given(text=valid_claim())` using Flask test client.
      - For each item in `resp.json["source_evidence"]`, assert all five keys present and `reliability`, `agreement` in `[0, 1]`.
    - `@settings(max_examples=100)` on each test.
    - _Requirements: 4.1, 4.3, 4.4, 5.2, 5.5, 5.7_

- [x] 5. Modify `frontend/app.js` — add Evidence Panel
  - [x] 5.1 Implement `renderEvidencePanel(d)` and insert into `render(d)`
    - Add a new function `renderEvidencePanel(d)` to `frontend/app.js`:
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
    - In `render(d)`, insert `box.append(renderEvidencePanel(d));` immediately after the `box.append(art, el("p", "legend", ...))` line and before the `const grid = el("div", "grid");` line. No other lines in `render(d)` change.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 5.2 Write property test for Evidence Panel link rendering (Property 14)
    - File: `backend/tests/test_similarity_properties.py` (add a note pointing to a companion JS test file; the actual test lives in `frontend/tests/test_evidence_panel.test.js` using fast-check + jsdom).
    - Create `frontend/tests/test_evidence_panel.test.js`:
      - Import `renderEvidencePanel` (extracted as a module export or tested via jsdom script injection).
      - Use `fc.array(evidenceItemArb, {minLength: 1})` where `evidenceItemArb` generates objects with `title`, `link`, `source_domain`, `reliability`, `agreement`.
      - `# Feature: source-grounded-verification, Property 14: all evidence links have target=_blank and rel=noopener`
      - For each `<a>` in the rendered panel, assert `a.target === "_blank"` and `a.rel === "noopener"`.
    - `{numRuns: 100}`
    - _Requirements: 6.1, 6.4_

- [x] 6. Checkpoint — End-to-end verification of modified `app.py` and `app.js`
  - Run all tests in `backend/tests/`. POST a real headline to `/api/analyze` (via Flask test client or `curl`) and confirm the response contains `source_evidence`, `confidence_flag`, and `source_signal`, and that all original fields are present. Ask the user if anything is unexpected.

- [x] 7. Create `backend/evaluate_grounded.py` — offline evaluation CLI
  - [x] 7.1 Implement the evaluation script with `--sample` and three-config reporting
    - Create `backend/evaluate_grounded.py`.
    - Accept CLI arguments: positional `csv_path` and optional `--sample N` (default: all rows, fixed seed `random.seed(42)` when sampling).
    - Load `BUNDLE`, `SOURCES`, `retriever`, `similarity` using the same paths as `app.py` (`ROOT / "models" / "model.joblib"`, `ROOT / "backend" / "sources.json"`).
    - For each row (`text`, `label`, optional `url`):
      1. Run `retriever.fetch(text)` → `similarity.compute_scores` → `similarity.compute_source_signal`.
      2. If retrieval returns empty due to network exception or RSS timeout, increment `fallback_count` and use ML-only score for all three configs.
      3. Compute predicted label for three configs: **ml_only** (existing `app.py` scoring without source signal), **source_only** (threshold `source_signal >= 0.5` → 1, else 0), **fused** (same fusion formula as the updated `app.py`).
    - Compute `accuracy`, `precision`, `recall`, `f1` for each config using `sklearn.metrics`.
    - Write `models/grounded_metrics.json` in the schema defined in `design.md` (fields: `dataset`, `rows`, `sample`, `fallback_count`, `results`).
    - Print a summary table to stdout.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 7.2 Write unit tests for `evaluate_grounded.py` in `backend/tests/test_evaluate_grounded.py`
    - Test: run script on a 5-row in-memory CSV with known labels and mocked `retriever.fetch` → `grounded_metrics.json` created with three `results` entries.
    - Test: when `retriever.fetch` raises an exception for one row → `fallback_count` equals 1 and that row still contributes to all three configs using ML-only score.
    - Test: `--sample 3` on a 10-row CSV → `sample` field in output JSON equals 3.
    - _Requirements: 10.3, 10.4, 10.5_

- [x] 8. Final checkpoint — Full test suite and latency smoke test
  - Run the complete test suite (`pytest backend/tests/`). Confirm `grounded_metrics.json` is generated correctly by running `evaluate_grounded.py` on a 50-row sample of `data/ifnd.csv`. Verify that a cached `/api/analyze` call returns within 300 ms (Requirement 8.1). Ask the user if anything needs attention before closing out.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; core functionality does not depend on them.
- Property tests require `hypothesis` (`pip install hypothesis`). Frontend property tests require `fast-check` (`npm install fast-check --save-dev`) with a jsdom environment.
- Each property test file uses `@settings(max_examples=100)` (Hypothesis) or `{numRuns: 100}` (fast-check).
- The `related_articles()` refactor in task 4.3 is a one-line body replacement with no signature change, so no other call sites in `app.py` need updating.
- `SIMILARITY_MODE` defaults to `"lightweight"`. Set `SIMILARITY_MODE=semantic` only if `sentence-transformers` is installed; the module falls back automatically if the import fails.
- All new backend files should be placed in `backend/`; test files in `backend/tests/` (create the directory and an empty `__init__.py` if it does not exist).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "1.2", "1.3", "1.4"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3"] },
    { "id": 5, "tasks": ["4.4", "4.5", "5.1"] },
    { "id": 6, "tasks": ["5.2", "7.1"] },
    { "id": 7, "tasks": ["7.2"] }
  ]
}
```
