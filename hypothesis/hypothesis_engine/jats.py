"""JATS, the XML PubMed Central articles come in: one article's text, licence and PMCID."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator

# kept out of the text: table cells, formulas' markup, pictures, labels like "Figure 1", and
# the reference list and footnotes (Europe PMC puts them in <body> as sections of their own)
_SKIP = {"table", "tex-math", "graphic", "inline-graphic", "media", "label", "alt-text", "object-id"}
_SKIP |= {"ref-list", "fn-group"}
_SKIP_SECTIONS = {"ref-list", "fn-group"}
_SKIP_MARKERS = {"fn", "aff", "author-notes", "table-fn"}  # footnote and affiliation symbols
# blocks that can sit inside a paragraph; they become paragraphs of their own
_BLOCKS = {"p", "list", "fig", "table-wrap", "disp-quote", "boxed-text", "def-list", "statement"}
_UNWANTED_ABSTRACTS = {"graphical", "toc", "teaser"}


def _skipped(element: ET.Element) -> bool:
    # namespaced elements are MathML and the like
    return (
        element.tag in _SKIP
        or element.tag.startswith("{")
        or (element.tag == "sec" and element.get("sec-type") in _SKIP_SECTIONS)
        or (element.tag == "xref" and element.get("ref-type") in _SKIP_MARKERS)
    )


def _inline_text(element: ET.Element) -> str:
    """The element's own text, without the blocks nested in it."""
    pieces = [element.text or ""]
    for child in element:
        if not (_skipped(child) or child.tag in _BLOCKS):
            pieces.append(_inline_text(child))
        pieces.append(child.tail or "")
    return " ".join("".join(pieces).split())


def _paragraphs(element: ET.Element) -> Iterator[str]:
    """Section titles and paragraphs under `element`, in reading order."""
    for child in element:
        if _skipped(child):
            continue
        if child.tag in ("p", "title"):
            text = _inline_text(child)
            if text:
                yield text
            if child.tag == "p":  # a list or figure inside the paragraph comes after it
                yield from _paragraphs(child)
        else:
            yield from _paragraphs(child)


def jats_text(article: ET.Element) -> str | None:
    """The `<article>`'s readable text: title, abstract, then the body by section, one paragraph per
    blank-line-separated block. The reference list, footnotes and <back> matter are left out;
    citation numbers stay where the text has them.

    None when the XML holds no body text (some copies are only a scanned PDF).
    """
    main = article.find("body")
    if main is None or not any(_inline_text(p) for p in main.iter("p")):
        return None
    body_parts = list(_paragraphs(main))
    parts: list[str] = []
    meta = article.find("front/article-meta")
    if meta is not None:
        title = meta.find("title-group/article-title")
        if title is not None and _inline_text(title):
            parts.append(_inline_text(title))
        for abstract in meta.findall("abstract"):
            if abstract.get("abstract-type") not in _UNWANTED_ABSTRACTS:
                parts.extend(_paragraphs(abstract))
    return "\n\n".join([*parts, *body_parts])


def licence_in(article: ET.Element) -> str | None:
    """The licence the `<article>` itself states, for a copy the lookup gave none for.

    In order: a licence URL, the licence's own words, its type, and for older papers with no
    `<license>` (BioMed Central's before about 2004) the copyright statement holding the terms.
    """
    permissions = article.find("front/article-meta/permissions")
    if permissions is None:
        return None
    for element in permissions.iter():
        if element.tag.endswith("license_ref") and element.text:
            return element.text.strip()
    licence = permissions.find("license")
    if licence is not None:
        href = next((v for k, v in licence.attrib.items() if k.endswith("href")), None)
        words = " ".join(filter(None, (_inline_text(p) for p in licence.findall("license-p"))))
        return href or words or licence.get("license-type")
    statement = permissions.find("copyright-statement")
    return (_inline_text(statement) or None) if statement is not None else None


def pmcid_of(article: ET.Element) -> str | None:
    """The article's PMCID ("PMC1234567"), as `<article-id pub-id-type="pmcid">` gives it."""
    for element in article.findall("front/article-meta/article-id"):
        kind, value = element.get("pub-id-type"), (element.text or "").strip()
        if kind == "pmcid" and value:
            return value
        if kind in ("pmc", "pmcaid") and value.isdigit():
            return f"PMC{value}"
    return None


def articles(body: bytes) -> dict[str, ET.Element]:
    """The articles in one PMC efetch answer (`<pmc-articleset>`), by PMCID.

    Raises `xml.etree.ElementTree.ParseError` on XML that cannot be read.
    """
    root = ET.fromstring(body)
    found = [root] if root.tag == "article" else root.findall("article")
    return {pmcid: article for article in found if (pmcid := pmcid_of(article))}
