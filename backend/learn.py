


"""Live learning: collect new labelled examples while the site runs, then retrain.

Two sources go into data/live.csv (columns: text,label,origin,ts):
  feedback  a user clicked "Actually real / Actually fake" on a result   (weight 3, both classes)
  rss       fresh headlines from sources.json domains rated >= 0.90      (weight 1, REAL class only)

Why it is built this way (good viva material):
  - The model NEVER labels its own predictions and learns from them: that would reinforce its own mistakes.
  - RSS headlines only add 'real' examples, so training uses class_weight='balanced' and
    harvest() caps RSS rows to avoid drowning the fake class.
  - live.csv goes into TRAINING only; the test set stays static, so accuracy stays comparable.

CLI:  python backend/learn.py harvest   |   retrain   |   status
      python backend/learn.py export [out.csv] [feedback|rss]   -> clean CSV copy of data/live.csv
      (default output is exports/live_export.csv, deliberately OUTSIDE data/ because train.py
       reads every data/*.csv and would otherwise count the same rows twice)"""
import csv, json, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data" / "live.csv"
SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())
FIELDS = ["text", "label", "origin", "ts"]
MAX_RSS_ROWS = 3000
_lock = threading.Lock()
state = dict(running=False, last="never", error=None)

def _append(rows):
    with _lock:
        LIVE.parent.mkdir(exist_ok=True)
        new = not LIVE.exists()
        with LIVE.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new: w.writeheader()
            w.writerows(rows)

def _existing():
    if not LIVE.exists(): return set(), 0, 0
    with LIVE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["text"] for r in rows}, len(rows), sum(r["origin"] == "rss" for r in rows)

def add_feedback(text: str, label: int) -> bool:
    """label: 0 = fake, 1 = real (what the USER says is the truth)."""
    text = " ".join(text.split())
    if len(text.split()) < 5 or label not in (0, 1): return False
    seen, _, _ = _existing()
    if text in seen: return False
    _append([dict(text=text, label=label, origin="feedback", ts=time.strftime("%Y-%m-%d %H:%M"))])
    return True

def harvest(per_source: int = 15, min_score: float = 0.90) -> int:
    """Pull recent headlines from highly reliable domains (label = real). Needs internet."""
    import feedparser, requests
    seen, _, n_rss = _existing()
    rows = []
    for domain, score in SOURCES.items():
        if score < min_score or n_rss + len(rows) >= MAX_RSS_ROWS: continue
        try:
            r = requests.get("https://news.google.com/rss/search", timeout=6,
                             params=dict(q=f"site:{domain} when:2d", hl="en-IN", gl="IN", ceid="IN:en"))
            for e in feedparser.parse(r.content).entries[:per_source]:
                title = e.title.rsplit(" - ", 1)[0].strip()      # Google News appends ' - Source'
                if len(title.split()) >= 5 and title not in seen:
                    seen.add(title)
                    rows.append(dict(text=title, label=1, origin="rss", ts=time.strftime("%Y-%m-%d %H:%M")))
        except Exception as ex:
            print(f"skip {domain}: {ex}")
    if rows: _append(rows)
    return len(rows)

def retrain():
    """Retrain on static + live data. Returns True on success; the web app then reloads the model."""
    import train
    state.update(running=True, error=None)
    try:
        train.main(verbose=False)
        state.update(last=time.strftime("%Y-%m-%d %H:%M"))
        return True
    except Exception as ex:
        state.update(error=str(ex)); return False
    finally:
        state["running"] = False

def export(out: Path = None, origin: str = None) -> int:
    """Write the live data to a CSV (columns text,label,origin,ts), optionally only one origin."""
    out = Path(out) if out else ROOT / "exports" / "live_export.csv"
    if not LIVE.exists(): return 0
    with LIVE.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not origin or r["origin"] == origin]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return len(rows)

def status() -> dict:
    _, total, rss = _existing()
    return dict(live_rows=total, rss_rows=rss, feedback_rows=total - rss, **state)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "harvest": print("added", harvest(), "headlines")
    elif cmd == "retrain": print("ok" if retrain() else f"failed: {state['error']}")
    elif cmd == "export":
        args = sys.argv[2:]
        org = args.pop() if args and args[-1] in ("feedback", "rss") else None
        export(args[0] if args else None, org)
    else: print(status())