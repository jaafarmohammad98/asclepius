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
import re
from html import escape
from typing import Any

from asclepius.interventions import COMPARISON as INTERVENTION_COMPARISON
from asclepius.interventions import SHARES
from asclepius.run import PROVISIONAL
from asclepius.selfreport import LABEL as SELF_REPORT_LABEL
from asclepius.web.findings import COUNTS, FEW, INTERVENTION_COUNTS, SHOWN_COLUMNS, STATISTICS, ZONES
from asclepius.web.style import head as page_head

ZONE_COLUMNS = tuple(f"zone_{z}" for z in ZONES)
# an `intervention` row's own columns: the intervention outcomes ranking as the page shows it
INTERVENTION_COLUMNS = ("rank", *INTERVENTION_COUNTS, *SHARES, "safety_signal", "not_ranked")
# `row` is `finding`, `intervention` for an intervention of the ranking, or `statement` for a
# section with neither, so every step is in the file
CSV_COLUMNS = (
    "row",
    "section",
    "statement",
    *SHOWN_COLUMNS,
    *ZONE_COLUMNS,
    *INTERVENTION_COLUMNS,
    "labels",
    "tool",
    "dataset_fingerprint",
    "run_date",
    "settings",
)
FINDING, INTERVENTION, STATEMENT = "finding", "intervention", "statement"
INTEGERS = frozenset((*COUNTS, *ZONE_COLUMNS, "rank", *INTERVENTION_COUNTS))
FLOATS = frozenset((*STATISTICS, *SHARES))
BOOLEANS = frozenset(("safety_signal",))


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
    ("nothing reliable yet" when it is, spec S8) and the versions and settings of `record` (S7). The
    intervention outcomes ranking has one `intervention` row per intervention, its counts and
    shares as shown. A section with neither, or that did not finish, has one `statement` row
    saying why."""
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
        for shown in section.get("ranking", []):
            row = {"row": INTERVENTION, **head, "comparison": INTERVENTION_COMPARISON}
            row |= {
                "marker_id": _intervention_id(shown["intervention"]),
                "marker_name": shown["intervention"],
            }
            row |= {c: shown.get(c) for c in INTERVENTION_COLUMNS}
            rows.append(row | {"labels": "; ".join(labels_of(section)), **run})
        if not section["findings"] and not section.get("ranking"):
            rows.append({"row": STATEMENT, **head, "labels": "; ".join(labels_of(section)), **run})
    return rows


def _intervention_id(name: str) -> str:
    """The id `asclepius.interventions` gives an intervention: its name, spaces as underscores."""
    return "intervention:" + name.replace(" ", "_")


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
    if column in BOOLEANS:
        return "true" if value else "false"
    return value


def read_findings_csv(text: str) -> list[dict[str, object]]:
    """A findings CSV read back: counts as int, statistics as float, flags as bool, empty cells as None."""
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
            elif column in BOOLEANS:
                row[column] = cell == "true"
            else:
                row[column] = cell
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------- HTML report

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
                f'<rect class="box side-{i}" x="{x(s["q25"])}" y="{y - 6}" '
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
                    body.append(f'<text class="on-{z}" x="{at + 3:.2f}" y="{y + 10}">{s[z]}</text>')
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


def _figures(cell: str) -> bool:
    """Whether a cell (escaped HTML) leads with figures: a count, a value, a share, or one hidden. They
    are set in the number face, right-aligned, as on the page."""
    text = re.sub(r"(?<=\d)e[+-]?(?=\d)", "", cell.split("<", 1)[0].strip())  # 1.14e+03 is a figure
    return text in (FEW, "hidden") or (not re.search(r"[A-Za-z]", text) and bool(re.search(r"[\d–]", text)))


def _cell_class(i: int, cell: str, text_columns: int) -> str:
    if cell.startswith("<svg"):
        return "plot"
    return "num" if i >= text_columns and _figures(cell) else "text"


def _table(head: list[str], rows: list[list[str]], text_columns: int = 1) -> str:
    """A table of already-escaped cells; the first `text_columns` are text, and a column of figures
    is right-aligned in the number face."""
    classes = [[_cell_class(i, c, text_columns) for i, c in enumerate(r)] for r in rows]
    figure_columns = {i for i in range(len(head)) if rows and all(c[i] == "num" for c in classes)}
    th = "".join(
        f'<th class="{"num" if i in figure_columns else "text"}">{h}</th>' for i, h in enumerate(head)
    )
    body = "".join(
        "<tr>" + "".join(f'<td class="{k}">{c}</td>' for c, k in zip(r, kinds, strict=True)) + "</tr>"
        for r, kinds in zip(rows, classes, strict=True)
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


def _strength(label: object) -> str:
    """A finding's strength as the page tags it."""
    kind = "description" if label == "description, not a test" else str(label or "unlabelled")
    return f'<span class="strength {_e(kind)}">{_e(label or "not labelled")}</span>'


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
                _strength(f.get("label"))
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


def _hidden_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Intervention rows with a hidden count as "hidden": it may be 5 or more, hidden beside one
    below 5, so "fewer than 5" would be wrong."""
    counts = (
        "n_authors",
        "n_clear",
        "major_improvement",
        "partial_improvement",
        "no_effect",
        "worse",
        "unclear",
    )
    return [{k: "hidden" if k in counts and v is None else v for k, v in r.items()} for r in rows]


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
        parts.append(_generic_table(_hidden_counts(s["ranking"])))
        parts.append("<h3>Safety signals: a high share reporting worse</h3>")
        parts.append(_generic_table(_hidden_counts(s.get("safety", []))) or '<p class="muted">None.</p>')
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
        page_head("Asclepius report", landscape=True) + f"<h1>Asclepius report</h1>{_chips(labels)}{head}"
        '<p class="muted">Counts below 5 are hidden and no single case is shown. The thresholds and '
        "the ratio list were chosen without a clinician and are provisional until expert review.</p>"
        f"{overall}{sections}"
        '<h2>Run record</h2><p class="muted">What this report was computed from, so it can be '
        "reproduced: the dataset's file hashes (not their names), the tool version and every setting.</p>"
        f"<pre>{_e(record_json)}</pre>"
        f'<script type="application/json" id="run-record">{embedded}</script>'
        "</body></html>\n"
    )
