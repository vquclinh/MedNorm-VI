"""Self-contained HTML report writer.

No external JS/CSS/fonts/CDNs/analytics — everything is inline. All
user-controlled text is HTML-escaped. The report is deterministic apart from the
explicitly rendered timestamp (isolated in the banner).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from . import EvaluationOutcome
from .models import EvaluationRunMetadata, PerEntityScore

_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;
 padding:0 0 4rem;line-height:1.45;background:#fff;color:#111}
@media(prefers-color-scheme:dark){body{background:#12141a;color:#e8e8ea}
 td,th{border-color:#333!important}.card{background:#1b1e26!important}}
.banner{background:#b3261e;color:#fff;padding:14px 20px;font-weight:700;
 font-size:1.25rem;letter-spacing:.06em;text-align:center}
.sub{background:#7a1a15;color:#ffd9d6;padding:6px 20px;text-align:center;font-size:.8rem}
.wrap{max-width:1100px;margin:0 auto;padding:20px}
h2{border-bottom:2px solid #b3261e;padding-bottom:4px;margin-top:2rem}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.card{background:#f5f5f7;border-radius:10px;padding:14px 18px;min-width:150px;flex:1}
.card .n{font-size:1.6rem;font-weight:700}
.card .l{font-size:.75rem;text-transform:uppercase;opacity:.7;letter-spacing:.05em}
table{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:.85rem}
th,td{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top}
th{background:rgba(179,38,30,.12)}
.scroll{overflow-x:auto}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.bad{color:#b3261e;font-weight:600}
.meta{font-size:.8rem;opacity:.8}
.bar{height:9px;background:#b3261e;border-radius:5px}
.barbg{background:#ddd;border-radius:5px;overflow:hidden;min-width:80px}
@media(prefers-color-scheme:dark){.barbg{background:#333}}
"""


def _pct(x: float) -> str:
    return f"{x:.4f}"


def _bar(x: float) -> str:
    w = max(0.0, min(1.0, x)) * 100.0
    return f'<div class="barbg"><div class="bar" style="width:{w:.1f}%"></div></div>'


def _score_cards(outcome: EvaluationOutcome) -> str:
    c = outcome.corpus
    cards = [
        ("Final score", c.final_score),
        ("Text (0.3)", c.text_score),
        ("Assertions (0.3)", c.assertions_score),
        ("Candidates (0.4)", c.candidates_score),
    ]
    html = ['<div class="grid">']
    for label, val in cards:
        html.append(f'<div class="card"><div class="n">{_pct(val)}</div>'
                    f'<div class="l">{escape(label)}</div></div>')
    html.append("</div>")
    return "".join(html)


def _kv_table(rows: list[tuple[str, str]]) -> str:
    out = ['<table><tbody>']
    for k, v in rows:
        out.append(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _group_table(title: str, groups: dict[str, dict[str, float]]) -> str:
    if not groups:
        return f"<p class='meta'>No {escape(title)} data.</p>"
    out = [f"<h2>{escape(title)}</h2><div class='scroll'><table><thead><tr>"
           "<th>Group</th><th>Final</th><th>Text</th><th>Assertions</th>"
           "<th>Candidates</th><th>Slots</th></tr></thead><tbody>"]
    for name in sorted(groups):
        g = groups[name]
        out.append(
            f"<tr><td>{escape(name)}</td>"
            f"<td>{_pct(g['final'])} {_bar(g['final'])}</td>"
            f"<td>{_pct(g['text'])}</td><td>{_pct(g['assertions'])}</td>"
            f"<td>{_pct(g['candidates'])}</td><td>{int(g['n_slots'])}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _worst_documents(outcome: EvaluationOutcome, k: int = 10) -> str:
    docs = sorted(outcome.per_document, key=lambda d: (d.final_score, d.document_id))[:k]
    out = ["<h2>Worst documents</h2><div class='scroll'><table><thead><tr>"
           "<th>Doc</th><th>Prov</th><th>Final</th><th>Text</th><th>Assert</th>"
           "<th>Cand</th><th>GT</th><th>Pred</th><th>Miss</th><th>Spur</th>"
           "</tr></thead><tbody>"]
    for d in docs:
        prov = d.provenance.value if d.provenance else "-"
        out.append(
            f"<tr><td>{escape(d.document_id)}</td><td>{escape(prov)}</td>"
            f"<td>{_pct(d.final_score)}</td><td>{_pct(d.text_score)}</td>"
            f"<td>{_pct(d.assertions_score)}</td><td>{_pct(d.candidates_score)}</td>"
            f"<td>{d.n_gt}</td><td>{d.n_pred}</td>"
            f"<td class='bad'>{d.n_missing}</td><td class='bad'>{d.n_spurious}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _diagnostics_table(outcome: EvaluationOutcome) -> str:
    counts = outcome.diagnostics.counts
    if not counts:
        return "<h2>Error categories</h2><p class='meta'>None.</p>"
    out = ["<h2>Error categories</h2><table><thead><tr><th>Category</th><th>Count</th>"
           "</tr></thead><tbody>"]
    for cat, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"<tr><td class='mono'>{escape(cat)}</td><td>{n}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _matched_table(slots: list[PerEntityScore], limit: int = 60) -> str:
    matched = [s for s in slots if s.slot_kind == "matched"][:limit]
    out = ["<h2>Matched entities</h2><div class='scroll'><table><thead><tr>"
           "<th>Doc</th><th>Type</th><th>GT text</th><th>Pred text</th>"
           "<th>GT pos</th><th>Pred pos</th><th>Text</th><th>Assert Δ</th>"
           "<th>Cand Δ</th></tr></thead><tbody>"]
    for s in matched:
        p = s.pair
        assert p is not None
        a_delta = ""
        if s.assertions is not None:
            a_delta = f"-{list(s.assertions.missing)} +{list(s.assertions.extra)}"
        c_delta = ""
        if s.candidates is not None:
            c_delta = f"-{list(s.candidates.missing)} +{list(s.candidates.extra)}"
        out.append(
            f"<tr><td>{escape(p.document_id)}</td><td>{escape(p.entity_type)}</td>"
            f"<td>{escape(p.gt_text)}</td><td>{escape(p.pred_text)}</td>"
            f"<td class='mono'>{escape(str(list(p.gt_position)))}</td>"
            f"<td class='mono'>{escape(str(list(p.pred_position)))}</td>"
            f"<td>{_pct(s.text_score)}</td>"
            f"<td class='mono'>{escape(a_delta)}</td>"
            f"<td class='mono'>{escape(c_delta)}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _unmatched_table(slots: list[PerEntityScore]) -> str:
    missing = [s for s in slots if s.slot_kind == "missing"]
    spurious = [s for s in slots if s.slot_kind == "spurious"]

    def _rows(items: list[PerEntityScore]) -> str:
        r = []
        for s in items[:80]:
            r.append(f"<tr><td>{escape(s.document_id)}</td>"
                     f"<td>{escape(s.entity_type)}</td>"
                     f"<td class='mono'>{escape(str(list(s.diagnostics)))}</td></tr>")
        return "".join(r)

    return (
        "<h2>Unmatched ground truth (missing)</h2><div class='scroll'><table><thead><tr>"
        "<th>Doc</th><th>Type</th><th>Diagnostics</th></tr></thead><tbody>"
        f"{_rows(missing)}</tbody></table></div>"
        "<h2>Unmatched predictions (spurious)</h2><div class='scroll'><table><thead><tr>"
        "<th>Doc</th><th>Type</th><th>Diagnostics</th></tr></thead><tbody>"
        f"{_rows(spurious)}</tbody></table></div>"
    )


def render_html(outcome: EvaluationOutcome, run_metadata: EvaluationRunMetadata,
                ground_truth_provenance: str) -> str:
    cfg = outcome.config
    meta_rows = [
        ("Data provenance", ground_truth_provenance),
        ("Matching strategy", cfg.matching_strategy),
        ("WER tokenization", cfg.tokenization),
        ("Aggregation policy", cfg.aggregation_policy),
        ("Text clipping", "enabled" if cfg.clipping_enabled else "disabled (raw preserved)"),
        ("Evaluator version", cfg.evaluator_version),
        ("Documents scored", str(outcome.corpus.n_documents)),
    ]
    body = [
        '<div class="banner">PROVISIONAL LOCAL EVALUATOR</div>',
        '<div class="sub">Not an official evaluator clone &middot; organizer test set has '
        'NO ground truth &middot; scores are provisional</div>',
        '<div class="wrap">',
        f'<p class="meta">Generated {escape(run_metadata.timestamp_utc)} &middot; '
        f'Python {escape(run_metadata.python_version)} &middot; '
        f'{escape(run_metadata.platform)}</p>',
        "<h2>Final provisional score</h2>",
        _score_cards(outcome),
        "<h2>Run configuration &amp; provenance</h2>",
        _kv_table(meta_rows),
        _group_table("Score by entity type", outcome.corpus.per_type),
        _group_table("Score by routed case (C1-C7)", outcome.corpus.per_case),
        _diagnostics_table(outcome),
        _worst_documents(outcome),
        _matched_table(list(outcome.slots)),
        _unmatched_table(list(outcome.slots)),
        "</div>",
    ]
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>MedNorm-VI — Provisional Local Evaluator</title>"
        f"<style>{_CSS}</style></head><body>{''.join(body)}</body></html>"
    )


def write_html(report_dir: str | Path, outcome: EvaluationOutcome,
               run_metadata: EvaluationRunMetadata, ground_truth_provenance: str) -> Path:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.html"
    path.write_text(render_html(outcome, run_metadata, ground_truth_provenance), encoding="utf-8")
    return path


__all__ = ["render_html", "write_html"]
