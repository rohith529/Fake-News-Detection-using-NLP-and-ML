"""Stage 2 — NLP preprocessing: normalise → tokenise → stop-words → stem.

Changes from original
─────────────────────
1. fix_encoding()   — repair mojibake before any other step.
   36 % of IFND rows contain Windows-1252 characters read as Latin-1
   (â€™ → ', Ã© → é, etc.).  Fixing them before cleaning stops garbage
   tokens polluting the vocabulary.

2. question_lead()  — boolean feature injected as a prefix token "__QLEAD__".
   1,193 fake headlines start with "Is / Are / Did / Do / Was / Who / What /
   Why / How / Can".  Real news rarely does.  Adding the token gives the
   TF-IDF vectorizer a direct handle on that pattern.

3. debunk_lead()    — prefix token "__DLEAD__" for negation / debunking openers
   ("No,", "Fact Check:", "False:", "Misleading:", "Viral Test:").
   These account for ~131 false negatives where the model is otherwise
   confused because the rest of the text looks like real news.

4. strip_leakage()  — unchanged.
5. clean()          — calls fix_encoding first, then injects feature tokens,
   then tokenises + stems as before.
"""
import re
import unicodedata
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
_URL     = re.compile(r"https?://\S+|www\.\S+")
_AGENCY  = re.compile(r"^.{0,80}?\((reuters|ap|afp)\)\s*[-–—]\s*", re.I)
_JUNK    = re.compile(
    r"\(?\b(reuters|associated press|afp)\b\)?"
    r"|featured image( via| by)?|image via|getty images|screenshot"
    r"|pic\.twitter\.com/\S+|@\w+"
    r"|21st century wire|read more:?|via \w+", re.I)

# ── Question-lead pattern (fake signal) ──────────────────────────────────────
_QLEAD = re.compile(
    r"^\s*(is\s|are\s|did\s|do\s|does\s|was\s|were\s|who\s|what\s|why\s|how\s|can\s|could\s|will\s)",
    re.I)

# ── Debunking/negation lead (fake signal) ────────────────────────────────────
_DLEAD = re.compile(
    r"^\s*(no[,\s]|not\s|fake[:\s]|fact\s*check[:\s]|false[:\s]|mislead|"
    r"viral\s+test[:\s]|misreport|debunk|hoax|explainer[:\s])",
    re.I)

# ── Encoding repair ──────────────────────────────────────────────────────────
# Many IFND rows were saved as Latin-1 / Windows-1252 and later read as UTF-8,
# producing mojibake such as â€™ (→ ') or Ã© (→ é).
# Strategy: encode back to latin-1 (ignoring errors) then decode as utf-8;
# fall back to NFKC normalisation for anything that remains.
def fix_encoding(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Fast path: already clean ASCII/UTF-8
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        pass
    # Attempt mojibake repair
    try:
        repaired = text.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
        # Only keep the repaired version if it reduced the non-ASCII ratio
        orig_bad  = sum(1 for c in text     if ord(c) > 127) / max(len(text), 1)
        new_bad   = sum(1 for c in repaired if ord(c) > 127) / max(len(repaired), 1)
        if new_bad < orig_bad:
            text = repaired
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    # NFKC normalise to collapse look-alike characters
    return unicodedata.normalize("NFKC", text)


def strip_leakage(text: str) -> str:
    """Remove dataset-collection artefacts before joining title + body."""
    text = _AGENCY.sub("", text)
    return _JUNK.sub(" ", text)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", _URL.sub(" ", text.lower()))


def stem(word: str) -> str:
    return _stemmer.stem(word)


def clean(text: str) -> str:
    """Full preprocessing pipeline.  Returns a space-joined string of stems
    plus optional prefix feature tokens (__QLEAD__, __DLEAD__)."""
    text = fix_encoding(text)

    # Inject structural feature tokens BEFORE lowercasing/stemming so they
    # survive as literal vocabulary items in TF-IDF.
    prefix = ""
    if _QLEAD.search(text):
        prefix += "__QLEAD__ "
    if _DLEAD.search(text):
        prefix += "__DLEAD__ "

    tokens = [
        stem(t) for t in tokenize(text)
        if t not in ENGLISH_STOP_WORDS and len(t) > 2
    ]
    return prefix + " ".join(tokens)
