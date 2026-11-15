"""Explainable AI: word-level contributions + linguistic red-flag markers."""
import re
from preprocess import clean, stem

SENSATIONAL = {"shocking", "unbelievable", "miracle", "exposed", "bombshell",
               "hoax", "outrage", "conspiracy", "hidden", "explosive", "whistleblower", "cabal",
               "coverup", "plandemic", "illuminati", "deepstate"}
CLICKBAIT = ["you won't believe", "doctors hate", "share before", "what happens next",
             "before it gets deleted", "they don't want you", "media is hiding", "click now",
             "you need to see this", "this will shock you", "banned by", "they are hiding"]
# Negation-lead debunking patterns common in IFND fake headlines
_NEGATION = re.compile(
    r"^(no[,\s]|not\s|fake[:\s]|fact\s*check[:\s]|false[:\s]|misleading[:\s]|"
    r"viral\s+test[:\s]|is\s+this\s+real|did\s+.*really|who\s+are\s+these)", re.I)

def linguistic_markers(text: str) -> dict:
    words = re.findall(r"[A-Za-z']+", text)
    n = max(len(words), 1)
    caps = sum(1 for w in words if len(w) > 2 and w.isupper()) / n
    excl = text.count("!")
    low = text.lower()
    sens = sorted({w for w in re.findall(r"[a-z]+", low) if w in SENSATIONAL})
    bait = [p for p in CLICKBAIT if p in low]
    negation = bool(_NEGATION.search(text.strip()))
    flags = []
    if caps > 0.1: flags.append(f"{caps:.0%} of words are ALL CAPS")
    if excl: flags.append(f"{excl} exclamation mark(s)")
    if sens: flags.append("sensational words: " + ", ".join(sens))
    if bait: flags.append("clickbait phrases: " + "; ".join(bait))
    if negation: flags.append("negation/debunking lead — common in fake-news headlines")
    if n < 8: flags.append("very short text - low evidence")
    # Clickbait phrases are a stronger signal; negation lead is a moderate signal
    score = min(1.0, 2 * caps + 0.15 * excl + 0.15 * len(sens) + 0.5 * len(bait) + (0.4 if negation else 0))
    return dict(score=round(score, 2), flags=flags)

def explain_words(text: str, vec, lr, top: int = 6) -> dict:
    """contribution = LR coefficient x TF-IDF weight. Positive -> real, negative -> fake."""
    row = vec.transform([clean(text)]).tocoo()
    contrib = {j: lr.coef_[0][j] * v for j, v in zip(row.col, row.data)}
    vocab, names = vec.vocabulary_, vec.get_feature_names_out()
    peak = max((abs(c) for c in contrib.values()), default=1) or 1
    tokens = []
    for piece in re.findall(r"\w+|\W+", text):
        s = 0.0
        if piece[0].isalnum():
            idx = vocab.get(stem(piece.lower()))
            s = contrib.get(idx, 0.0) / peak if idx is not None else 0.0
        tokens.append(dict(t=piece, s=round(float(s), 2)))
    ranked = sorted(contrib.items(), key=lambda kv: kv[1])
    return dict(tokens=tokens,
                fake_terms=[names[j] for j, c in ranked[:top] if c < 0],
                real_terms=[names[j] for j, c in ranked[::-1][:top] if c > 0])
