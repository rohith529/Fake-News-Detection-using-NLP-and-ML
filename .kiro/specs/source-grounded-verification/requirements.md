# Requirements Document

## Introduction

This feature extends the existing Veritas Desk fake news detector with source-grounded verification. The current system scores credibility using only a TF-IDF + Linear SVM classifier trained on historical headlines (IFND dataset). That classifier cannot verify claims about recent events, produces no supporting evidence, and can be fooled by well-written fake articles that mimic real writing style.

The source-grounded verification feature adds a second, complementary signal: the system retrieves live articles from curated trusted sources and measures how well those articles agree with or contradict the submitted claim. The two signals — ML classifier and live source agreement — are fused into a single credibility score. Every result now shows the retrieved evidence articles so the user can judge for themselves.

The feature must work inside the existing Flask + vanilla-JS stack using only libraries already installed (scikit-learn, numpy, requests, feedparser). Installing `sentence-transformers` is a deliberate design choice that must be evaluated against a lighter TF-IDF cosine-similarity approach; the requirements capture both paths as an optional-feature requirement.

## Glossary

- **Veritas_Desk**: The existing fake news detection web application (Flask backend + vanilla-JS frontend).
- **Claim**: A news headline, article text, or URL submitted by the user for credibility assessment.
- **Retriever**: The backend component that fetches live articles from trusted sources for a given Claim.
- **Source_Corpus**: The set of articles returned by the Retriever for a given Claim; used as the evidence base.
- **Similarity_Engine**: The backend component that computes a numerical agreement score between a Claim and each article in the Source_Corpus.
- **Agreement_Score**: A floating-point value in [0, 1] indicating how closely a retrieved article supports the Claim. 0 = no overlap/contradiction detected, 1 = strong textual alignment.
- **Source_Signal**: The weighted aggregate Agreement_Score across the Source_Corpus for a given Claim; a value in [0, 1].
- **ML_Signal**: The existing TF-IDF + classifier probability that the Claim is real; unchanged from the current system.
- **Fusion_Engine**: The backend component that combines ML_Signal and Source_Signal into the final Credibility_Score.
- **Credibility_Score**: An integer in [0, 100] representing the fused probability that the Claim is real.
- **Verdict**: The categorical label derived from Credibility_Score: "Likely Real" (≥ 60), "Likely Fake" (< 40), or "Unverified" (40–59). Thresholds match the existing CSS gradient bands.
- **Evidence_Panel**: The UI section that displays retrieved source articles alongside their Agreement_Scores and links.
- **Trusted_Domain**: A news domain listed in `sources.json` with a reliability score ≥ 0.80.
- **Lightweight_Similarity**: TF-IDF cosine similarity computed with scikit-learn on stemmed tokens; uses no additional dependencies.
- **Semantic_Similarity**: Dense-vector cosine similarity computed with `sentence-transformers` (all-MiniLM-L6-v2 or equivalent); requires installing the library (~90 MB download).
- **Cache**: An in-process LRU store that maps a Claim's first 12 words to its Source_Corpus; avoids repeat network calls within the same Flask process lifetime.
- **Confidence_Flag**: A boolean attached to the Source_Signal indicating whether at least two independent Trusted_Domain articles were retrieved; used to decide whether the Source_Signal influences the Verdict.

---

## Requirements

### Requirement 1: Retrieve Live Articles for Every Submitted Claim

**User Story:** As a user, I want the system to automatically find live news articles related to my claim, so that the credibility verdict is grounded in current reporting rather than only historical training data.

#### Acceptance Criteria

1. WHEN a Claim is submitted to the `/api/analyze` endpoint, THE Retriever SHALL query Google News RSS using the first 10–12 significant words of the Claim as the search query.
2. WHEN the RSS query returns results, THE Retriever SHALL filter results to retain only articles whose `source` domain matches a Trusted_Domain in `sources.json`.
3. THE Retriever SHALL return at most 8 articles per Claim submission.
4. WHEN the RSS request does not complete within 5 seconds, THE Retriever SHALL return an empty Source_Corpus and set the Confidence_Flag to false.
5. WHEN the Source_Corpus for a Claim is already in the Cache, THE Retriever SHALL return the cached Source_Corpus without issuing a new network request.
6. IF the Retriever encounters an exception during fetching, THEN THE Retriever SHALL log the exception message and return an empty Source_Corpus without raising an error to the caller.

---

### Requirement 2: Compute Agreement Score Between Claim and Retrieved Articles

**User Story:** As a developer, I want a Similarity_Engine that numerically measures how well each retrieved article agrees with the submitted claim, so that the system can distinguish supporting from non-supporting coverage.

#### Acceptance Criteria

1. THE Similarity_Engine SHALL compute an Agreement_Score for each article in the Source_Corpus against the Claim.
2. WHEN computing Agreement_Score, THE Similarity_Engine SHALL apply the same preprocessing pipeline (`backend/preprocess.py` `clean()`) to both the Claim text and the article title before comparison.
3. THE Similarity_Engine SHALL produce Agreement_Scores in the range [0, 1] where higher values indicate greater textual alignment.
4. WHERE Lightweight_Similarity is configured, THE Similarity_Engine SHALL compute Agreement_Score as the cosine similarity between TF-IDF vectors fitted on the Claim and Source_Corpus texts using scikit-learn's `TfidfVectorizer`.
5. WHERE Semantic_Similarity is configured and `sentence-transformers` is installed, THE Similarity_Engine SHALL compute Agreement_Score as the cosine similarity between sentence embeddings produced by the configured model.
6. WHEN the Source_Corpus contains zero articles, THE Similarity_Engine SHALL return an empty list of Agreement_Scores and set the Confidence_Flag to false.
7. FOR ALL Claim inputs of at least 5 words, THE Similarity_Engine SHALL produce a deterministic Agreement_Score (same input → same score) when Lightweight_Similarity is configured.

---

### Requirement 3: Compute Source Signal from Agreement Scores

**User Story:** As a developer, I want the system to aggregate individual Agreement_Scores into a single Source_Signal, so that the Fusion_Engine has a stable, normalised input.

#### Acceptance Criteria

1. THE Similarity_Engine SHALL compute the Source_Signal as the reliability-weighted mean of the top-3 Agreement_Scores from the Source_Corpus, where each article's weight equals its source domain's reliability score from `sources.json`.
2. WHEN fewer than 3 articles are returned, THE Similarity_Engine SHALL compute the weighted mean over all available articles.
3. WHEN the Source_Corpus contains zero articles, THE Similarity_Engine SHALL set the Source_Signal to 0.5 (neutral) and set the Confidence_Flag to false.
4. WHEN fewer than 2 Trusted_Domain articles are retrieved, THE Similarity_Engine SHALL set the Confidence_Flag to false.
5. WHEN at least 2 Trusted_Domain articles are retrieved, THE Similarity_Engine SHALL set the Confidence_Flag to true.

---

### Requirement 4: Fuse ML Signal and Source Signal into a Final Credibility Score

**User Story:** As a user, I want the final credibility score to reflect both the ML classifier's learned patterns and live source agreement, so that neither signal alone determines the verdict.

#### Acceptance Criteria

1. WHEN the Confidence_Flag is true, THE Fusion_Engine SHALL compute the Credibility_Score as (0.55 × ML_Signal + 0.45 × Source_Signal) × 100, rounded to the nearest integer.
2. WHEN the Confidence_Flag is false, THE Fusion_Engine SHALL compute the Credibility_Score using the existing weighting logic in `app.py` (ML_Signal and linguistic markers only), and SHALL NOT apply the 0.45 Source_Signal weight.
3. THE Fusion_Engine SHALL clamp the Credibility_Score to the range [0, 100].
4. THE Fusion_Engine SHALL derive the Verdict from the Credibility_Score using these exclusive thresholds: Credibility_Score ≥ 60 → "Likely Real"; Credibility_Score < 40 → "Likely Fake"; 40 ≤ Credibility_Score ≤ 59 → "Unverified". A Credibility_Score of exactly 40 SHALL map to "Unverified".
5. WHEN the linguistic marker score exceeds 0.70 (strong clickbait signal), THE Fusion_Engine SHALL apply the existing marker override logic regardless of the Source_Signal value.

---

### Requirement 5: Expose Source Evidence in the API Response

**User Story:** As a user, I want to see which trusted articles the system retrieved and how strongly they agree with my claim, so that I can evaluate the evidence rather than trusting a black-box score.

#### Acceptance Criteria

1. THE `/api/analyze` endpoint SHALL include a `source_evidence` field in every JSON response.
2. WHEN the Source_Corpus contains at least one article, THE `source_evidence` field SHALL contain an array of objects, each with the keys: `title` (string), `link` (string), `source_domain` (string), `reliability` (number in [0, 1]), and `agreement` (number in [0, 1], rounded to 2 decimal places).
3. WHEN the Source_Corpus is empty, THE `source_evidence` field SHALL contain an empty array.
4. WHEN the Source_Corpus is non-empty but no article entries can be parsed (e.g. all entries lack a title or link field), THE `/api/analyze` endpoint SHALL treat this case identically to an empty Source_Corpus: `source_evidence` SHALL be an empty array and `confidence_flag` SHALL be false.
5. THE `/api/analyze` endpoint SHALL include a `confidence_flag` boolean field in every JSON response.
6. THE `/api/analyze` endpoint SHALL include a `source_signal` field (number in [0, 1], rounded to 3 decimal places) in every JSON response.
7. THE `/api/analyze` response SHALL preserve all existing fields (`verdict`, `credibility`, `p_real`, `model`, `latency_ms`, `markers`, `explanation`, `source`, `related`, `factchecks`) so that existing frontend code continues to function without modification.

---

### Requirement 6: Display Source Evidence in the Frontend

**User Story:** As a user, I want a dedicated Evidence Panel in the results view that shows the retrieved source articles and their agreement scores, so that I can click through to primary sources.

#### Acceptance Criteria

1. WHEN the API response contains a non-empty `source_evidence` array, THE Evidence_Panel SHALL render each article as a clickable link with its title, source domain name, and agreement score displayed as a percentage.
2. WHEN the API response contains an empty `source_evidence` array, THE Evidence_Panel SHALL display the message "No live coverage found — verdict based on ML classifier only."
3. WHEN the `confidence_flag` is false, THE Evidence_Panel SHALL display a notice: "Low source coverage — treat the score with caution."
4. THE Evidence_Panel SHALL open article links in a new browser tab with `rel="noopener"`.
5. WHEN the Credibility_Score changes from the previous result to the new result, THE Evidence_Panel SHALL update its content without requiring a page reload.
6. THE Evidence_Panel SHALL be rendered after the existing word-explanation section and before the fact-check links section in the results view.

---

### Requirement 7: Similarity Engine Configuration

**User Story:** As a developer, I want to switch between the lightweight TF-IDF similarity and the heavier semantic similarity at startup without changing application code, so that the system can run in both low-resource and high-accuracy modes.

#### Acceptance Criteria

1. THE Similarity_Engine SHALL read a `SIMILARITY_MODE` environment variable at Flask startup.
2. WHEN `SIMILARITY_MODE` is set to `"semantic"` and `sentence-transformers` is importable, THE Similarity_Engine SHALL use Semantic_Similarity for all Agreement_Score computations.
3. WHEN `SIMILARITY_MODE` is set to `"semantic"` and either `sentence-transformers` is not importable or the model fails to initialize (e.g. missing weights, download failure), THE Similarity_Engine SHALL catch the exception, emit a warning message to `stderr` including the failure reason, and fall back to Lightweight_Similarity for all Agreement_Score computations during that process lifetime.
4. WHEN `SIMILARITY_MODE` is absent or set to any value other than `"semantic"`, THE Similarity_Engine SHALL use Lightweight_Similarity.
5. THE Similarity_Engine SHALL log the active similarity mode as an `INFO`-level message at Flask startup.

---

### Requirement 8: Response Latency Under Source Retrieval

**User Story:** As a user, I want analysis results to appear within a reasonable time even when live article retrieval is slow, so that the UI remains usable.

#### Acceptance Criteria

1. WHEN Lightweight_Similarity is active and the Source_Corpus is served from the Cache, THE `/api/analyze` endpoint SHALL respond within 300 ms.
2. WHEN Lightweight_Similarity is active and a live RSS fetch is required, THE `/api/analyze` endpoint SHALL respond within 6 seconds (5 s retrieval timeout + 1 s processing budget).
3. WHEN Semantic_Similarity is active and a live RSS fetch is required, THE `/api/analyze` endpoint SHALL respond within 10 seconds.
4. IF the total analysis time exceeds 10 seconds, THEN THE `/api/analyze` endpoint SHALL return the result computed from ML_Signal and linguistic markers only, with an empty `source_evidence` array and `confidence_flag` set to false.

---

### Requirement 9: Similarity Engine Round-Trip Consistency

**User Story:** As a developer, I want to verify that the similarity computation is internally consistent, so that the same claim always produces the same Agreement_Score when given the same corpus.

#### Acceptance Criteria

1. FOR ALL Claim texts of at least 5 words and Source_Corpus lists of at least 1 article, THE Similarity_Engine SHALL produce the same Agreement_Score list when called twice with identical inputs (determinism property).
2. WHEN a Claim text is compared against a Source_Corpus containing only that same text, THE Similarity_Engine SHALL return an Agreement_Score of 1.0 for that article (self-similarity identity property).
3. WHEN a Claim text is compared against a Source_Corpus containing a text with zero token overlap after preprocessing, THE Similarity_Engine SHALL return an Agreement_Score of 0.0 for that article (orthogonality property).
4. FOR ALL Source_Corpus lists, THE Similarity_Engine SHALL return Agreement_Scores in the same order as the input article list (order-preservation property).

---

### Requirement 10: Accuracy Evaluation on Labelled Claim Sample

**User Story:** As a developer, I want to measure whether source-grounded verification improves classification accuracy over the ML-only baseline, so that the academic report can include a quantitative comparison.

#### Acceptance Criteria

1. THE system SHALL provide an offline evaluation script (`backend/evaluate_grounded.py`) that accepts a CSV file with columns `text`, `label` (0 = fake, 1 = real) and optional `url`.
2. WHEN the evaluation script is run, THE Evaluation_Script SHALL compute Accuracy, Precision, Recall, and F1-score for three configurations: ML-only baseline, Source-Signal-only, and Fused (ML + Source_Signal).
3. THE Evaluation_Script SHALL write results to `models/grounded_metrics.json` in the same schema as the existing `metrics.json`.
4. WHEN the Source_Corpus for a Claim cannot be retrieved due to a network exception or RSS timeout, THE Evaluation_Script SHALL score that row using ML-only scoring as a substitute for all three configurations (ML-only, Source-Signal-only, and Fused), include it in all metric calculations, and increment the `fallback_rows` counter in the output.
5. THE Evaluation_Script SHALL accept a `--sample N` argument to limit evaluation to N randomly sampled rows (default: all rows).
