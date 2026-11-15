const $ = s => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

$("#ex1").onclick = () => { $("#text").value = "SHOCKING: Scientists confirm hot water cures all cancers, doctors hate this!"; $("#url").value = ""; };
$("#ex2").onclick = () => { $("#text").value = "Central bank keeps interest rates unchanged, citing stable inflation and steady economic growth"; $("#url").value = "https://www.thehindu.com/business/"; };

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = $("#go"); btn.disabled = true; btn.textContent = "Checking...";
  try {
    const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("#text").value, url: $("#url").value }) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed");
    render(d); loadHistory();
  } catch (err) {
    const box = $("#result"); box.hidden = false; box.replaceChildren(el("p", "err", err.message));
  } finally { btn.disabled = false; btn.textContent = "Check credibility"; }
});

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

function render(d) {
  const box = $("#result"); box.hidden = false; box.replaceChildren();

  const v = el("div", "verdict");
  const verdictClass = d.verdict.replace(/ /g, "-");
  const strong = el("strong", verdictClass, d.verdict);
  const scoreLabel = `credibility ${d.credibility}/100 (model p_real=${Math.round(d.p_real*100)}%)`;
  v.append(strong, el("span", "score", scoreLabel));
  const rule = el("div", "rule"), needle = el("div", "needle"); needle.style.left = "0%"; rule.append(needle);
  requestAnimationFrame(() => requestAnimationFrame(() => needle.style.left = d.credibility + "%"));
  // Tick marks at 40 and 60, matching the CSS gradient bands
  const ticks = el("div", "ticks");
  [["Fake", "0%"], ["", "40%"], ["Unverified", "50%"], ["", "60%"], ["Real", "100%"]].forEach(([t, pos]) => {
    const s = el("span", null, t); s.style.position = "absolute"; s.style.left = pos;
    s.style.transform = "translateX(-50%)"; ticks.append(s);
  });
  ticks.style.position = "relative";
  box.append(v, rule, ticks);

  box.append(el("h3", null, "Why: words that pushed the score"));
  const art = el("p", "article");
  d.explanation.tokens.forEach(tk => {
    const s = el("span", null, tk.t);
    if (Math.abs(tk.s) > 0.15) s.style.background = `color-mix(in srgb, var(${tk.s < 0 ? "--fake" : "--real"}) ${Math.round(Math.abs(tk.s) * 45)}%, transparent)`;
    art.append(s);
  });
  box.append(art, el("p", "legend", "Red words pushed toward fake, green toward real."));
  box.append(renderEvidencePanel(d));

  const grid = el("div", "grid");
  const m = el("div"); m.append(el("h3", null, "Language red flags"));
  const ul = el("ul", "flags");
  (d.markers.flags.length ? d.markers.flags : ["No sensational patterns found"]).forEach(f => ul.append(el("li", null, f)));
  m.append(ul);

  const s = el("div"); s.append(el("h3", null, "Source"));
  const src = d.source;
  s.append(el("p", null, !src.domain ? "No link given, so the source did not affect the score."
    : src.score == null ? `${src.domain}: not in the reliability list.`
    : `${src.domain}: reliability ${Math.round(src.score * 100)}/100.`));
  if (src.watchlist) s.append(el("p", "err", `Watchlist: flagged fake ${src.flagged} times here.`));
  s.append(el("p", "muted", `Model: ${d.model}, ${d.latency_ms} ms`));

  const terms = el("div"); terms.append(el("h3", null, "Strongest terms"));
  d.explanation.fake_terms.forEach(t => { const c = el("span", "chip", t); c.style.color = "var(--fake)"; terms.append(c); });
  d.explanation.real_terms.forEach(t => { const c = el("span", "chip", t); c.style.color = "var(--real)"; terms.append(c); });
  grid.append(m, s, terms); box.append(grid);

  const ev = el("div", "grid");
  const rel = el("div"); rel.append(el("h3", null, "Check against trusted coverage"));
  const rl = el("ul", "plain");
  if (d.related.length) d.related.forEach(a => { const li = el("li"); const l = el("a", null, a.title); l.href = a.link; l.target = "_blank"; l.rel = "noopener"; li.append(l, el("span", "muted", a.source ? ` - ${a.source}` : "")); rl.append(li); });
  else rl.append(el("li", "muted", d.verdict === "Likely real" ? "Not needed for a likely-real result." : "No related articles found (or you are offline)."));
  rel.append(rl);
  const fc = el("div"); fc.append(el("h3", null, "Fact-check sites"));
  const fl = el("ul", "plain");
  d.factchecks.forEach(f => { const li = el("li"); const l = el("a", null, f.name); l.href = f.link; l.target = "_blank"; l.rel = "noopener"; li.append(l); fl.append(li); });
  fc.append(fl); ev.append(rel, fc); box.append(ev);

  // Teach the model: a human label is the only trustworthy signal for live learning
  const fb = el("div", "row"), sent = $("#text").value;
  fb.append(el("span", "muted", "Teach the model - what is the truth? "));
  [["Actually real", 1], ["Actually fake", 0]].forEach(([name, label]) => {
    const b = el("button", "ghost", name); b.type = "button";
    b.onclick = async () => {
      const r = await (await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sent, label }) })).json();
      fb.replaceChildren(el("span", "muted", r.saved ? "Thanks - saved for the next retrain." : "Already saved (or text too short)."));
      loadLearn();
    };
    fb.append(b);
  });
  box.append(fb);
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadHistory() {
  const rows = await (await fetch("/api/history")).json();
  const ul = $("#history");
  if (!rows.length) return;
  ul.replaceChildren(...rows.map(r => { const li = el("li"); li.append(el("strong", r.verdict.replace(" ", "-"), r.verdict), el("span", null, ` ${r.credibility}/100 `), el("span", "muted", r.snippet)); return li; }));
}

async function loadStats() {
  const d = await (await fetch("/api/stats")).json();
  $("#meta").textContent = `Dataset: ${d.dataset} (${d.rows} rows + ${d.live_rows || 0} live; test set ${d.test_rows || "?"} rows, static only). Best model: ${d.best}.`;
  const t = $("#stats"), head = el("tr");
  ["Model", "Accuracy", "Precision", "Recall", "F1", "Confusion [[TN,FP],[FN,TP]]"].forEach(h => head.append(el("th", null, h)));
  t.replaceChildren(head);
  d.results.forEach(r => { const tr = el("tr"); [r.model, r.accuracy, r.precision, r.recall, r.f1, JSON.stringify(r.confusion)].forEach(c => tr.append(el("td", null, c))); t.append(tr); });
}
async function loadLearn() {
  const d = await (await fetch("/api/learn/status")).json();
  $("#learn").textContent = `Live data: ${d.feedback_rows} corrections + ${d.rss_rows} trusted headlines. Last retrain: ${d.last}.` +
    (d.running ? " Retraining..." : "") + (d.error ? ` Error: ${d.error}` : "");
  if (d.running) setTimeout(loadLearn, 3000); else loadStats();
}
$("#retrain").onclick = async () => {
  await fetch("/api/retrain", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  loadLearn();
};
loadHistory(); loadStats(); loadLearn();
