"""Exports of a web-app run, for papers and sharing (spec S7, S10, S11): the findings as CSV and a
report as HTML.

Both are made from the findings view (`asclepius.web.findings`), so they show exactly what the
page shows: no count below 5, no single case. Both name the tool version and the dataset version
(its fingerprint), and carry "provisional, needs expert review" and, on self-report findings,
"self-report, no controls".

The CSV has one row per finding. `read_findings_csv` reads it back to the values the view had:
counts as whole numbers, statistics as floats, a hidden or missing value as None.
"""

from __future__ import annotations

import csv
import io
import json
from html import escape
from typing import Any

from asclepius.run import PROVISIONAL
from asclepius.selfreport import LABEL as SELF_REPORT_LABEL
from asclepius.web.findings import COUNTS, FEW, SHOWN_COLUMNS, STATISTICS, ZONES

ZONE_COLUMNS = tuple(f"zone_{z}" for z in ZONES)
# `row` is `finding`, or `statement` for a section with no findings, so every step is in the file
CSV_COLUMNS = (
    "row",
    "section",
    "statement",
    *SHOWN_COLUMNS,
    *ZONE_COLUMNS,
    "labels",
    "tool",
    "dataset_fingerprint",
    "run_date",
    "settings",
)
FINDING, STATEMENT = "finding", "statement"
INTEGERS = frozenset((*COUNTS, *ZONE_COLUMNS))
FLOATS = frozenset(STATISTICS)


def labels_of(section: dict[str, Any], finding: dict[str, Any] | None = None) -> list[str]:
    """The labels that apply to a finding of `section` (or to the section): its own, and
    "provisional, needs expert review" always (spec, Provisional defaults)."""
    labels = list(section.get("labels", []))
    caveat = (finding or {}).get("caveat")
    if caveat and caveat not in labels:
        labels.append(caveat)
    if not any(PROVISIONAL in label for label in labels):
        labels.append(PROVISIONAL)
    return labels


def findings_rows(view: dict[str, Any], record: dict[str, Any]) -> list[dict[str, object]]:
    """One CSV row per finding of `view`, in the page's order, each with its section's statement
    ("nothing reliable yet" when it is, spec S8) and the versions and settings of `record` (S7). A
    section with no findings, or that did not finish, has one `statement` row saying why."""
    run = {
        "tool": record["tool"],
        "dataset_fingerprint": record["dataset"]["fingerprint"],
        "run_date": record.get("date"),
        "settings": json.dumps(record.get("settings", {}), sort_keys=True, allow_nan=False),
    }
    rows = []
    for section in view["sections"]:
        said = section.get("statement") if section["state"] == "done" else None
        said = said or f"{section['state']}: {section.get('message') or 'no findings'}"
        head = {"section": section["kind"], "statement": said}
        for finding in section["findings"]:
            row: dict[str, object] = {"row": FINDING, **head}
            row |= {c: finding.get(c) for c in SHOWN_COLUMNS}
            zones = finding.get("zones") or {}
            row |= {f"zone_{z}": zones.get(z) for z in ZONES}
            rows.append(row | {"labels": "; ".join(labels_of(section, finding)), **run})
        if not section["findings"]:
            rows.append({"row": STATEMENT, **head, "labels": "; ".join(labels_of(section)), **run})
    return rows


def findings_csv(view: dict[str, Any], record: dict[str, Any]) -> str:
    """The findings of `view` as CSV text; None is an empty cell, floats are written exactly."""
    out = io.StringIO()
    writer = csv.DictWriter(out, CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in findings_rows(view, record):
        writer.writerow({k: _cell(k, row.get(k)) for k in CSV_COLUMNS})
    return out.getvalue()


def _cell(column: str, value: object) -> object:
    if value is None:
        return ""
    if column in FLOATS:
        return repr(float(value))  # type: ignore[arg-type]  # repr gives the float back exactly
    return value


def read_findings_csv(text: str) -> list[dict[str, object]]:
    """A findings CSV read back: counts as int, statistics as float, empty cells as None."""
    rows = []
    for raw in csv.DictReader(io.StringIO(text)):
        row: dict[str, object] = {}
        for column, cell in raw.items():
            if cell == "":
                row[column] = None
            elif column in INTEGERS:
                row[column] = int(cell)
            elif column in FLOATS:
                row[column] = float(cell)
            else:
                row[column] = cell
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------- HTML report

STYLE = """
:root { color-scheme: light dark; --bg: #ffffff; --text: #1d1f21; --muted: #5f6368; --line: #dadce0;
  --accent: #2f5d8a; --ok: #1e6b3a; --warn: #7a5200; --bar: #e8eaed;
  --below: #4a7fb5; --within: #9aa0a6; --above: #c0723a; --unclear: #d5d8dc; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #17181a; --text: #e8eaed; --muted: #a0a4a8; --line: #3c4043; --accent: #8ab4f8;
    --ok: #81c995; --warn: #fdd663; --bar: #3c4043; --below: #7fa9d6; --within: #80868b;
    --above: #e39a66; --unclear: #5f6368; }
}
body { margin: 0 auto; max-width: 1200px; padding: 24px 16px; background: var(--bg); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.5em; margin: 0 0 4px; } h2 { font-size: 1.2em; margin: 28px 0 6px; }
h3 { font-size: 1em; margin: 14px 0 4px; }
.muted { color: var(--muted); }
.label { display: inline-block; border: 1px solid var(--warn); color: var(--warn); border-radius: 10px;
  padding: 0 8px; margin: 0 6px 4px 0; font-size: .85em; }
.nothing { border: 2px solid var(--warn); border-radius: 8px; padding: 10px 14px; font-weight: 600; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 6px 0 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: right; white-space: nowrap;
  vertical-align: middle; }
th:first-child, td:first-child, td.text, th.text { text-align: left; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; margin: 8px 0; }
dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
pre { background: var(--bar); padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; }
svg text { font-size: 11px; fill: var(--text); }
.whisker { stroke: var(--muted); stroke-width: 1.5; } .box { fill: var(--bar); stroke: var(--accent); }
.median { stroke: var(--accent); stroke-width: 2.5; }
.share-bg { fill: var(--bar); } .share { fill: var(--accent); }
.zone-below { fill: var(--below); } .zone-within { fill: var(--within); } .zone-above { fill: var(--above); }
.zone-unclear { fill: var(--unclear); } .hidden-part { fill: none; stroke: var(--muted); stroke-dasharray: 3 2; }
@media print { .scroll { overflow: visible; } }
"""
W, LABEL_W, ROW_H = 300, 132, 18


def _e(value: object) -> str:
    return escape("" if value is None else str(value))


def _count(n: object) -> str:
    return FEW if n is None else f"{n:,}"


def _num(x: object) -> str:
    return "–" if x is None else f"{x:.3g}"  # type: ignore[str-format]


def _pct(x: object) -> str:
    return "hidden" if x is None else f"{round(x * 100)}%"  # type: ignore[operator]


def _side_n(side: dict[str, Any]) -> str:
    """A plot side's size; "<5" (fewer than 5) where the label is short, as on the page."""
    return "<5" if side.get("n") is None else str(side["n"])


def _svg(rows: int, title: str, body: list[str], footer: bool = True) -> str:
    height = rows * ROW_H + (16 if footer else 2)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" role="img"><title>{_e(title)}</title>{"".join(body)}</svg>'
    )


def _label(i: int, side: dict[str, Any]) -> str:
    return f'<text x="0" y="{i * ROW_H + 13}">{_e(side["group"])} ({_e(_side_n(side))})</text>'


def distribution_svg(sides: list[dict[str, Any]]) -> str:
    """Per side: the 10th to 90th percentile as a line, the 25th to 75th as a box, the median."""
    title = "; ".join(
        f"{s['group']}: n {_count(s['n'])}, median {_num(s['median'])}, 25th–75th {_num(s['q25'])}–"
        f"{_num(s['q75'])}, 10th–90th {_num(s['q10'])}–{_num(s['q90'])}"
        for s in sides
    )
    body = [_label(i, s) for i, s in enumerate(sides)]
    drawn = [s for s in sides if s.get("median") is not None]
    if drawn:
        lo, hi = min(s["q10"] for s in drawn), max(s["q90"] for s in drawn)
        span, width = (hi - lo) or 1, W - LABEL_W - 4

        def x(v: float) -> float:
            return round(LABEL_W + (v - lo) / span * width, 2)

        for i, s in enumerate(sides):
            if s.get("median") is None:
                continue
            y = i * ROW_H + 9
            body += [
                f'<line class="whisker" x1="{x(s["q10"])}" x2="{x(s["q90"])}" y1="{y}" y2="{y}"/>',
                f'<rect class="box" x="{x(s["q25"])}" y="{y - 6}" '
                f'width="{max(1, x(s["q75"]) - x(s["q25"])):.2f}" height="12"/>',
                f'<line class="median" x1="{x(s["median"])}" x2="{x(s["median"])}" y1="{y - 7}" y2="{y + 7}"/>',
            ]
        y = len(sides) * ROW_H + 12
        body += [
            f'<text x="{LABEL_W}" y="{y}">{_e(_num(lo))}</text>',
            f'<text x="{W - 4}" y="{y}" text-anchor="end">{_e(_num(hi))}</text>',
        ]
    return _svg(len(sides), title, body)


def shares_svg(sides: list[dict[str, Any]], caption: str = "share who took the drug") -> str:
    """Per side: a share (that took the drug, or that reports a symptom), named by `caption`."""
    title = "; ".join(f"{s['group']}: {_pct(s['share'])}, n {_count(s['n'])}" for s in sides)
    width, body = W - LABEL_W - 40, []
    for i, s in enumerate(sides):
        y = i * ROW_H + 3
        body += [_label(i, s), f'<rect class="share-bg" x="{LABEL_W}" y="{y}" width="{width}" height="12"/>']
        if s.get("share") is not None:
            body.append(
                f'<rect class="share" x="{LABEL_W}" y="{y}" width="{s["share"] * width:.2f}" height="12"/>'
            )
        body.append(f'<text x="{LABEL_W + width + 4}" y="{y + 10}">{_pct(s.get("share"))}</text>')
    body.append(f'<text x="{LABEL_W}" y="{len(sides) * ROW_H + 12}">{_e(caption)}</text>')
    return _svg(len(sides), title, body)


def zones_svg(sides: list[dict[str, Any]]) -> str:
    """Per group: how many values fall below, within and above the range (hidden ones dashed)."""
    zones = [z for z in ZONES if z in sides[0]]
    title = "; ".join(
        f"{s['group']}: n {_count(s['n'])}; " + ", ".join(f"{z} {_count(s[z])}" for z in zones) for s in sides
    )
    width, body = W - LABEL_W - 4, []
    for i, s in enumerate(sides):
        y = i * ROW_H + 3
        body.append(_label(i, s))
        at = float(LABEL_W)
        if s.get("n"):
            for z in zones:
                if not s.get(z):  # hidden, or 0
                    continue
                w = s[z] / s["n"] * width
                body.append(f'<rect class="zone-{z}" x="{at:.2f}" y="{y}" width="{w:.2f}" height="12"/>')
                if w > 22:
                    body.append(f'<text x="{at + 3:.2f}" y="{y + 10}">{s[z]}</text>')
                at += w
        if at < LABEL_W + width - 0.5:
            body.append(
                f'<rect class="hidden-part" x="{at:.2f}" y="{y}" width="{LABEL_W + width - at:.2f}" height="12"/>'
            )
    return _svg(len(sides), title, body, footer=False)


PLOTS = {"distribution": distribution_svg, "shares": shares_svg, "zones": zones_svg}


def plot_svg(plot: dict[str, Any] | None) -> str:
    if not plot or plot.get("type") not in PLOTS or not plot.get("sides"):
        return ""
    if plot["type"] == "shares" and plot.get("caption"):
        return shares_svg(plot["sides"], str(plot["caption"]))
    return PLOTS[plot["type"]](plot["sides"])


def _table(head: list[str], rows: list[list[str]], text_columns: int = 1) -> str:
    """A table of already-escaped cells; the first `text_columns` are left-aligned text."""
    th = "".join(
        f'<th class="text">{h}</th>' if i < text_columns else f"<th>{h}</th>" for i, h in enumerate(head)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="text">{c}</td>' if i < text_columns else f"<td>{c}</td>" for i, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _marker(f: dict[str, Any]) -> str:
    name = _e(f.get("marker_name") or f.get("marker_id"))
    measure = f.get("measure") or f.get("comparison")
    return f'{name}<br><span class="muted">{_e(measure)}</span>' if measure else name


def _effect(f: dict[str, Any]) -> str:
    if f.get("effect_size") is None:
        return "–"
    effect = f"{_num(f['effect_size'])} [{_num(f.get('ci_low'))}, {_num(f.get('ci_high'))}]"
    if f.get("odds_ratio") is not None:
        effect += (
            f'<br><span class="muted">odds ratio {_num(f["odds_ratio"])} '
            f"[{_num(f.get('odds_ratio_low'))}, {_num(f.get('odds_ratio_high'))}]</span>"
        )
    return effect


def _tier_a(f: dict[str, Any]) -> str:
    result = _e(f.get("tier_a_result") or "–")
    if f.get("tier_a_effect_size") is None:
        return result
    return (
        f'{result}<br><span class="muted">n {_count(f.get("tier_a_n_a"))} / {_count(f.get("tier_a_n_b"))}, '
        f"{_num(f['tier_a_effect_size'])} [{_num(f.get('tier_a_ci_low'))}, {_num(f.get('tier_a_ci_high'))}]</span>"
    )


def _tested_table(findings: list[dict[str, Any]]) -> str:
    head = ["Marker", "Sex", "Groups", "n", "Medians", "Effect [95% CI]", "q", "Label", "Tier A", "Plot"]
    rows = []
    for f in findings:
        medians = f"{_num(f.get('median_a'))} / {_num(f.get('median_b'))}"
        if f.get("share_a") is not None or f.get("share_b") is not None:
            medians += (
                f'<br><span class="muted">took it {_pct(f.get("share_a"))} / {_pct(f.get("share_b"))}</span>'
            )
        rows.append(
            [
                _marker(f),
                _e(f.get("sex") or "–"),
                f'{_e(f.get("group_a"))}<br><span class="muted">vs {_e(f.get("group_b"))}</span>',
                f"{_count(f.get('n_a'))} / {_count(f.get('n_b'))}",
                medians,
                _effect(f),
                _num(f.get("q_value")),
                _e(f.get("label") or "not labelled")
                + (f'<br><span class="muted">{_e(f["caveat"])}</span>' if f.get("caveat") else ""),
                _tier_a(f),
                plot_svg(f.get("plot")),
            ]
        )
    return _table(head, rows, text_columns=3)


def _range_table(findings: list[dict[str, Any]]) -> str:
    head = ["Marker", "Sex", "Group", "n", "Below", "Within", "Above", "Unclear", "Median position", "Plot"]
    rows = []
    for f in findings:
        zones = f.get("zones") or {}
        rows.append(
            [
                _e(f.get("marker_name") or f.get("marker_id")),
                _e(f.get("sex") or "–"),
                _e(f.get("group_a")),
                _count(f.get("n_a")),
                *(_count(zones.get(z)) for z in ZONES),
                _num(f.get("median_position")),
                plot_svg(f.get("plot")),
            ]
        )
    return _table(head, rows, text_columns=3)


def _generic_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    head = list(rows[0])
    cells = [[_e(_generic_cell(k, r.get(k))) for k in head] for r in rows]
    return _table([_e(h.replace("_", " ")) for h in head], cells, text_columns=1)


def _generic_cell(key: str, value: object) -> object:
    """A summary cell: a hidden count says "fewer than 5" (spec S11), another missing value "–"."""
    if value is None:
        return FEW if key == "n" or key.startswith("n_") else "–"
    return _num(value) if isinstance(value, float) else value


def _refused(refused: list[dict[str, Any]]) -> str:
    if not refused:
        return ""
    head = ["Marker", "Sex", "Groups", "n", "Why not tested"]
    rows = [
        [
            _e(r.get("marker_name") or r.get("comparison")),
            _e(r.get("sex") or "–"),
            f"{_e(r.get('group_a'))} vs {_e(r.get('group_b'))}",
            f"{_count(r.get('n_a'))} / {_count(r.get('n_b'))}",
            _e(r.get("reason")),
        ]
        for r in refused
    ]
    return f"<h3>Not tested</h3>{_table(head, rows, text_columns=3)}"


def _chips(labels: list[str]) -> str:
    return "".join(f'<span class="label">{_e(label)}</span>' for label in labels)


def _section_html(s: dict[str, Any]) -> str:
    parts = [f'<h2>{_e(s["title"])} <span class="muted">({_e(s["state"])})</span></h2>', _chips(labels_of(s))]
    if s["state"] != "done":
        parts.append(
            f'<p class="muted">{_e(s.get("message") or "No findings: this step did not finish.")}</p>'
        )
        return "<section>" + "".join(parts) + "</section>"
    if s.get("nothing_reliable"):
        parts.append(
            f'<p class="nothing">Nothing reliable yet. <span class="muted">{_e(s.get("statement"))}</span></p>'
        )
    elif s.get("statement"):
        parts.append(f"<p>{_e(s['statement'])}</p>")
    if s.get("note"):
        parts.append(f'<p class="muted">{_e(s["note"])}</p>')
    clusters = s.get("clusters")
    if clusters:
        parts.append(f"<p>{_e(clusters.get('statement'))}</p>")
        if clusters.get("reason"):
            parts.append(f'<p class="muted">{_e(clusters["reason"])}</p>')
        sizes = [{"cluster": c["cluster"], "sex": c["sex"], "n": c["n"]} for c in clusters.get("sizes", [])]
        parts.append(_generic_table(sizes))
    if s.get("ranking"):
        parts.append("<h3>Reported outcomes, ranked by the share improved</h3>")
        parts.append(_generic_table(s["ranking"]))
        parts.append("<h3>Safety signals: a high share reporting worse</h3>")
        parts.append(_generic_table(s.get("safety", [])) or '<p class="muted">None.</p>')
    if s.get("symptoms"):
        parts.append("<h3>Most over-represented symptoms per cluster</h3>")
        parts.append(_generic_table(s["symptoms"]))
    if s.get("groups"):
        parts.append("<h3>Authors per drug class and drug</h3>")
        parts.append(_generic_table(s["groups"]))
    if s.get("summary"):
        parts.append(f"<h3>Pattern per marker (reference: {_e(s.get('reference'))})</h3>")
        parts.append(_generic_table(s["summary"]))
    findings = s.get("findings", [])
    if findings:
        ranges = all("zones" in f for f in findings)
        parts.append(_range_table(findings) if ranges else _tested_table(findings))
    parts.append(_refused(s.get("refused", [])))
    return "<section>" + "".join(parts) + "</section>"


def report_html(view: dict[str, Any], record: dict[str, Any]) -> str:
    """One self-contained HTML file: the run's versions and labels, each step's findings with
    their plots, and the run record, shown and embedded as JSON (`id="run-record"`)."""
    dataset = record["dataset"]
    labels = [PROVISIONAL]
    if view.get("kind") == "self-report" or view.get("label"):
        labels.insert(0, str(view.get("label") or SELF_REPORT_LABEL))
    record_json = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
    embedded = record_json.replace("<", "\\u003c")  # no "</script>" can end the block early
    groups = " vs ".join(record.get("groups", [])) or "–"
    head = (
        "<dl>"
        f"<dt>Tool</dt><dd>{_e(record['tool'])} (code {_e(str(record.get('code_sha256', ''))[:12])}…)</dd>"
        f"<dt>Dataset</dt><dd>{_e(dataset['kind'])}, fingerprint {_e(dataset['fingerprint'])}</dd>"
        f"<dt>Groups</dt><dd>{_e(groups)}</dd>"
        f"<dt>Run</dt><dd>{_e(record.get('date'))}</dd>"
        "</dl>"
    )
    overall = ""
    if view.get("nothing_reliable"):
        overall = (
            '<p class="nothing">Nothing reliable yet. <span class="muted">No finding in this run is '
            "strong or suggestive. That is a result, not an error.</span></p>"
        )
    sections = "".join(_section_html(s) for s in view["sections"])
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Asclepius report</title><style>{STYLE}</style></head><body>"
        f"<h1>Asclepius report</h1>{_chips(labels)}{head}"
        '<p class="muted">Counts below 5 are hidden and no single case is shown. The thresholds and '
        "the ratio list were chosen without a clinician and are provisional until expert review.</p>"
        f"{overall}{sections}"
        '<h2>Run record</h2><p class="muted">What this report was computed from, so it can be '
        "reproduced: the dataset's file hashes (not their names), the tool version and every setting.</p>"
        f"<pre>{_e(record_json)}</pre>"
        f'<script type="application/json" id="run-record">{embedded}</script>'
        "</body></html>\n"
    )
