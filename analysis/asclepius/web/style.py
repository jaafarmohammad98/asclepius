"""The look the HTML exports share with the app page: `almanac.css`, its fonts carried inside the file
as data so an export opened offline, or sent on, looks as the page does, and a layer of its own for a
report read on screen and printed on A4 or Letter."""

from __future__ import annotations

import base64
import re
from functools import cache
from pathlib import Path

WEB = Path(__file__).parent

REPORT = """
body { max-width: 1180px; margin: 0 auto; padding: var(--space-xl) var(--space-xl) 80px; }
body > h1 { font-size: 2.25rem; line-height: 1.05; margin: 0 0 var(--space-sm); }
body > h2 { font-size: var(--text-xl); margin: var(--space-xl) 0 var(--space-sm); }
body > .label { margin: 0 6px var(--space-sm) 0; }
body > p { max-width: 80ch; }
section, article { background: var(--color-sheet); border-top: 2px solid var(--color-ink);
  padding: var(--space-md) var(--space-lg) var(--space-lg); margin-top: var(--space-lg); }
section > h2, article > h2 { font-size: var(--text-lg); margin: 0 0 var(--space-xs); }
section > h2 .muted { font: 400 var(--text-sm)/1 var(--font-body); letter-spacing: 0; }
section h3 { font-size: var(--text-base); margin: var(--space-lg) 0 var(--space-xs); }
article h3 { font-size: var(--text-base); margin: var(--space-lg) 0 var(--space-xs); padding-top: var(--space-sm);
  border-top: 1px solid var(--color-rule); }
article ol, article ul, body > ul { padding-left: 22px; }
article li, body > ul > li { margin-bottom: var(--space-sm); }
dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); margin: var(--space-md) 0; border-top: 2px solid var(--color-rule-2);
  font-size: var(--text-sm); }
dl dt, dl dd { margin: 0; padding: 6px 0; border-bottom: 1px solid var(--color-rule); }
dl dt { color: var(--color-muted); padding-right: var(--space-lg); }
dl dd { color: var(--color-ink); overflow-wrap: anywhere; }
.scroll { overflow-x: auto; margin: var(--space-sm) 0 var(--space-md); }
td.text { min-width: 7em; }
td.plot { min-width: 310px; }
td.plot svg { display: block; }
pre { font: 400 0.78rem/1.5 var(--font-num); background: var(--color-sheet); border-top: 2px solid var(--color-ink);
  padding: var(--space-md); overflow-x: auto; color: var(--color-ink-2); }
@media (max-width: 900px) {
  body { padding: var(--space-lg) var(--space-md) 64px; }
  section, article { padding: var(--space-md); }
}
@page { margin: 14mm 12mm; }
@media print {
  :root { --color-paper: #fff; --color-sheet: #fff; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  html, body { overflow: visible; }
  body { max-width: none; padding: 0; font-size: 10pt; }
  body > h1 { font-size: 20pt; }
  section, article { padding: var(--space-sm) 0 0; margin-top: var(--space-lg); break-inside: auto; }
  h1, h2, h3 { break-after: avoid; }
  .scroll { overflow: visible; }
  table { font-size: 7.5pt; }
  th, td { padding: 4px 5px; }
  th, .label, .strength { white-space: normal; }
  table { table-layout: auto; width: 100%; }
  td.text { min-width: 0; }
  td.plot { min-width: 0; width: 150px; }
  td.plot svg { width: 150px; height: auto; }
  thead { display: table-header-group; }
  tr, blockquote, dl, .nothing, .notice { break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 7pt; border-top-width: 1px; }
}
"""

# the striped fill of an "unclear" zone in the plots (`svg .zone-unclear`), once per page
HATCH = (
    '<svg width="0" height="0" style="position: absolute" aria-hidden="true"><defs>'
    '<pattern id="almanac-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<rect width="4" height="4" fill="oklch(89.5% 0.012 245)"/><rect width="1" height="4" fill="oklch(82% 0.013 240)"/>'
    "</pattern></defs></svg>"
)


def _data(path: Path, kind: str) -> str:
    return f"data:{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"


@cache
def export_style() -> str:
    """`almanac.css` with its fonts inlined as data, then the report's own layer."""
    css = (WEB / "almanac.css").read_text()

    def inline(match: re.Match[str]) -> str:
        return f'url("{_data(WEB / match.group(1), "font/woff2")}")'

    css = re.sub(r'url\("(vendor/fonts/[a-z0-9-]+\.woff2)"\)', inline, css)
    return css + REPORT


@cache
def favicon() -> str:
    return f'<link rel="icon" type="image/svg+xml" href="{_data(WEB / "favicon.svg", "image/svg+xml")}">'


def head(title: str, landscape: bool = False) -> str:
    """An export's document head and the start of its body: the shared look, the icon, the plots' fill.
    `landscape` prints it across the page, for wide tables."""
    page = "@page { size: A4 landscape; }" if landscape else ""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>{favicon()}<style>{export_style()}{page}</style></head><body>{HATCH}"
    )
