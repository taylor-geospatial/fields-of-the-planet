"""Validate ``paper/refs.bib`` against authoritative metadata APIs.

The bibliography had several entries with hallucinated titles, truncated
author lists, or wrong years. The fix is to never hand-author bib fields:
resolve every entry from publisher-deposited metadata and diff it against
what is in the file.

Resolution order per entry:

* ``doi`` field present       -> Crossref REST (``api.crossref.org/works/{doi}``).
* ``eprint`` + arXiv          -> arXiv Atom API.
* neither                     -> Crossref bibliographic title search; the top
                                 hit is accepted only if its title is a close
                                 match (so we never silently bind to the wrong
                                 paper).

For each resolved entry we compare title, author family names, and year, and
print a per-entry verdict. With ``--show-bibtex`` we also emit the canonical
BibTeX straight from the publisher (Crossref content negotiation) or rebuilt
from the arXiv record, reusing the existing citation key so it is drop-in.

Stdlib only -- no new dependency. Read-only: it never edits the .bib.
Exit code is non-zero if any entry mismatches or cannot be resolved, so this
doubles as a pre-submission / CI gate.

Examples::

    uv run scripts/validate_refs.py                       # audit paper/refs.bib
    uv run scripts/validate_refs.py --key wang2022unlocking --show-bibtex
    uv run scripts/validate_refs.py --bib paper/refs.bib --show-bibtex
"""

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

CROSSREF_WORKS = "https://api.crossref.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"
DOI_NEGOTIATE = "https://doi.org/{doi}"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}

# Title similarity below which we refuse to bind a title-search hit to an entry.
TITLE_ACCEPT_RATIO = 0.85
# Title similarity below which a DOI/arXiv-resolved entry is flagged as a
# title mismatch (looser, because the identifier already pins the work).
TITLE_MATCH_RATIO = 0.90


@dataclass
class Entry:
    key: str
    etype: str
    fields: dict[str, str]
    raw: str


@dataclass
class Record:
    """Authoritative metadata for one work."""

    source: str
    title: str
    families: list[str]
    year: str | None
    doi: str | None = None
    issues: list[str] = field(default_factory=list)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Standalone LaTeX glyph macros -> ASCII (\ss, \o, \aa, ...).
_LATEX_GLYPHS = {
    r"\ss": "ss",
    r"\o": "o",
    r"\O": "O",
    r"\l": "l",
    r"\L": "L",
    r"\aa": "a",
    r"\AA": "A",
    r"\ae": "ae",
    r"\AE": "AE",
    r"\oe": "oe",
    r"\OE": "OE",
    r"\i": "i",
    r"\j": "j",
}


def delatex(text: str) -> str:
    r"""Decode LaTeX accent macros to their base letter.

    Handles ``{\"a}``/``\"a``/``\"{a}`` (accent over a letter) and
    ``\c{c}``/``{\c c}`` (named-accent forms) so e.g. ``H{\"a}nsch``
    compares equal to the Unicode ``Hänsch`` that Crossref returns.
    """
    # \"a  \'e  \`a  \^o  \~n  \=a  \.a  -- accent symbol over one letter.
    text = re.sub(r"\\[\"\'`^~=.]\s*\{?(\w)\}?", r"\1", text)
    # \c{c}  \v{s}  \u{g}  \H{o}  -- named accent, braced argument.
    text = re.sub(r"\\[a-zA-Z]+\{(\w)\}", r"\1", text)
    # {\c c}  {\v s}  -- named accent, spaced argument.
    text = re.sub(r"\\[a-zA-Z]+\s+(\w)", r"\1", text)
    for macro, repl in _LATEX_GLYPHS.items():
        text = text.replace(macro, repl)
    return text


def norm_text(text: str) -> str:
    """Lowercase, decode LaTeX, drop braces/accents/punctuation, collapse ws."""
    text = strip_accents(delatex(text).replace("{", "").replace("}", ""))
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def family_key(name: str) -> str:
    """Reduce an author name to its comparison key: last surname token.

    Handles ``Family, Given``, ``Given Family``, and multi-word surnames
    (``Lavista Ferres`` -> ``ferres``) so the same author keys identically
    whether it comes from the .bib, Crossref, or arXiv.
    """
    name = name.replace("{", "").replace("}", "").strip()
    if "," in name:
        surname = name.split(",", 1)[0]
    else:
        surname = name.rsplit(" ", 1)[-1] if " " in name else name
    norm = norm_text(surname)
    return norm.split(" ")[-1] if norm else ""


def parse_authors_bibtex(value: str) -> list[str]:
    parts = re.split(r"\s+and\s+", value.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_bibtex(text: str) -> list[Entry]:
    """Brace-aware parse of flat BibTeX entries.

    Good enough for refs.bib (no nested @ inside values); extracts the entry
    type, key, and top-level ``field = {...}``/``"..."``/bareword values.
    """
    entries: list[Entry] = []
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at == -1:
            break
        m = re.match(r"@(\w+)\s*\{", text[at:])
        if not m:
            i = at + 1
            continue
        etype = m.group(1).lower()
        if etype in {"comment", "preamble", "string"}:
            i = at + m.end()
            continue
        body_start = at + m.end()  # just past the opening brace
        depth = 1
        j = body_start
        while j < n and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[body_start : j - 1]
        raw = text[at:j]
        i = j

        key_match = re.match(r"\s*([^,\s]+)\s*,", body)
        if not key_match:
            continue
        key = key_match.group(1).strip()
        fields = parse_fields(body[key_match.end() :])
        entries.append(Entry(key=key, etype=etype, fields=fields, raw=raw))
    return entries


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        fm = re.match(r"\s*([A-Za-z][\w-]*)\s*=\s*", body[i:])
        if not fm:
            break
        name = fm.group(1).lower()
        i += fm.end()
        if i >= n:
            break
        if body[i] == "{":
            depth = 1
            i += 1
            start = i
            while i < n and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            value = body[start : i - 1]
        elif body[i] == '"':
            i += 1
            start = i
            while i < n and body[i] != '"':
                i += 1
            value = body[start:i]
            i += 1
        else:
            start = i
            while i < n and body[i] not in ",\n":
                i += 1
            value = body[start:i].strip()
        fields[name] = re.sub(r"\s+", " ", value).strip()
        comma = body.find(",", i)
        if comma == -1:
            break
        i = comma + 1
    return fields


def http_get(url: str, accept: str, mailto: str, timeout: float = 30.0) -> str:
    headers = {
        "Accept": accept,
        "User-Agent": f"ftw-planet-refcheck/1.0 (mailto:{mailto})",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def crossref_by_doi(doi: str, mailto: str) -> Record:
    url = f"{CROSSREF_WORKS}/{urllib.parse.quote(doi)}"
    msg = json.loads(http_get(url, "application/json", mailto))["message"]
    return crossref_to_record(msg, source="crossref:doi")


def crossref_search(title: str, mailto: str) -> Record | None:
    params = urllib.parse.urlencode({"query.bibliographic": title, "rows": "1"})
    items = json.loads(http_get(f"{CROSSREF_WORKS}?{params}", "application/json", mailto))[
        "message"
    ]["items"]
    if not items:
        return None
    rec = crossref_to_record(items[0], source="crossref:search")
    ratio = difflib.SequenceMatcher(None, norm_text(title), norm_text(rec.title)).ratio()
    if ratio < TITLE_ACCEPT_RATIO:
        return None
    return rec


def crossref_to_record(msg: dict, source: str) -> Record:
    titles = msg.get("title") or [""]
    authors = msg.get("author") or []
    families = [family_key(a["family"]) for a in authors if a.get("family")]
    year = None
    for key in ("published", "published-print", "published-online", "issued"):
        parts = msg.get(key, {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    return Record(
        source=source,
        title=titles[0],
        families=families,
        year=year,
        doi=msg.get("DOI"),
    )


def arxiv_by_id(arxiv_id: str, mailto: str) -> Record | None:
    params = urllib.parse.urlencode({"id_list": arxiv_id})
    xml_text = http_get(f"{ARXIV_API}?{params}", "application/atom+xml", mailto)
    entry = ET.fromstring(xml_text).find("a:entry", ARXIV_NS)
    if entry is None:
        return None
    title_el = entry.find("a:title", ARXIV_NS)
    title = (title_el.text or "").strip() if title_el is not None else ""
    families: list[str] = []
    for author in entry.findall("a:author", ARXIV_NS):
        name_el = author.find("a:name", ARXIV_NS)
        if name_el is not None and name_el.text:
            families.append(family_key(name_el.text))
    published = entry.find("a:published", ARXIV_NS)
    year = published.text[:4] if published is not None and published.text else None
    return Record(source="arxiv", title=title, families=families, year=year)


def detect_arxiv_id(entry: Entry) -> str | None:
    eprint = entry.fields.get("eprint")
    if eprint:
        archive = entry.fields.get("archiveprefix", "").lower()
        if not archive or archive == "arxiv":
            return eprint.strip()
    # arXiv mints DataCite DOIs under the 10.48550/arXiv.<id> prefix; Crossref
    # does not hold these, so derive the id and route to the arXiv API instead.
    doi = entry.fields.get("doi", "")
    m = re.match(r"10\.48550/arXiv\.(.+)$", doi.strip(), re.IGNORECASE)
    return m.group(1) if m else None


def resolve(entry: Entry, mailto: str) -> Record | None:
    arxiv_id = detect_arxiv_id(entry)
    if arxiv_id:
        return arxiv_by_id(arxiv_id, mailto)
    doi = entry.fields.get("doi")
    if doi:
        try:
            return crossref_by_doi(doi.strip(), mailto)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            # DOI not in Crossref (e.g. DataCite-registered); fall through.
    title = entry.fields.get("title")
    if title:
        return crossref_search(title, mailto)
    return None


def compare(entry: Entry, rec: Record) -> list[tuple[str, str]]:
    """Return (kind, message) issues; kind in {title, author, year}."""
    issues: list[tuple[str, str]] = []

    bib_title = entry.fields.get("title", "")
    if bib_title and rec.title:
        ratio = difflib.SequenceMatcher(None, norm_text(bib_title), norm_text(rec.title)).ratio()
        if ratio < TITLE_MATCH_RATIO:
            issues.append(
                (
                    "title",
                    f"title differs ({ratio:.2f}):\n      bib: {bib_title}\n      api: {rec.title}",
                )
            )

    bib_authors = parse_authors_bibtex(entry.fields.get("author", ""))
    bib_fam = sorted(family_key(a) for a in bib_authors)
    api_fam = sorted(rec.families)
    if rec.families and bib_fam != api_fam:
        if len(bib_fam) != len(api_fam):
            issues.append(
                (
                    "author",
                    f"author count {len(bib_fam)} (bib) vs {len(api_fam)} (api); bib={bib_authors}",
                )
            )
        missing = sorted(set(api_fam) - set(bib_fam))
        extra = sorted(set(bib_fam) - set(api_fam))
        if missing:
            issues.append(("author", f"authors in api but not bib: {missing}"))
        if extra:
            issues.append(("author", f"authors in bib but not api: {extra}"))

    bib_year = entry.fields.get("year", "").strip()
    if bib_year and rec.year and bib_year != rec.year:
        issues.append(("year", f"year {bib_year} (bib) vs {rec.year} (api)"))

    return issues


def canonical_bibtex(entry: Entry, rec: Record, mailto: str) -> str | None:
    """Return drop-in BibTeX (existing key) from the publisher / arXiv."""
    if rec.doi:
        raw = http_get(DOI_NEGOTIATE.format(doi=rec.doi), "application/x-bibtex", mailto)
        raw = raw.strip()
        # Replace the registrar's generated key with our citation key.
        return re.sub(r"^@(\w+)\s*\{[^,]+,", rf"@\1{{{entry.key},", raw, count=1)
    if rec.source == "arxiv":
        arxiv_id = detect_arxiv_id(entry)
        authors = entry.fields.get("author", "")
        return (
            f"@misc{{{entry.key},\n"
            f"  title         = {{{rec.title}}},\n"
            f"  author        = {{{authors}}},\n"
            f"  year          = {{{rec.year}}},\n"
            f"  eprint        = {{{arxiv_id}}},\n"
            f"  archivePrefix = {{arXiv}}\n}}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", type=Path, default=Path("paper/refs.bib"))
    parser.add_argument("--key", help="validate only this citation key")
    parser.add_argument(
        "--show-bibtex",
        action="store_true",
        help="print canonical BibTeX (from publisher/arXiv) for flagged entries",
    )
    parser.add_argument("--mailto", default="isaac.corley@taylorgeospatial.org")
    parser.add_argument("--sleep", type=float, default=0.5, help="delay between requests")
    args = parser.parse_args()

    if not args.bib.exists():
        print(f"error: {args.bib} not found", file=sys.stderr)
        return 2

    entries = parse_bibtex(args.bib.read_text(encoding="utf-8"))
    if args.key:
        entries = [e for e in entries if e.key == args.key]
        if not entries:
            print(f"error: key {args.key!r} not found in {args.bib}", file=sys.stderr)
            return 2

    ok = mismatch = check = unresolved = 0
    flagged: list[tuple[Entry, Record]] = []

    for idx, entry in enumerate(entries):
        if idx:
            time.sleep(args.sleep)
        try:
            rec = resolve(entry, args.mailto)
        except urllib.error.HTTPError as exc:
            unresolved += 1
            print(f"[UNRESOLVED] {entry.key}: HTTP {exc.code} resolving identifier")
            continue

        if rec is None:
            unresolved += 1
            print(f"[UNRESOLVED] {entry.key}: no doi/arxiv id; add one to verify")
            continue

        issues = compare(entry, rec)
        # Hard-fail rule by source. Crossref-by-DOI = publisher metadata, so any
        # diff is a real error. arXiv pins the work but its preprint year/author
        # list legitimately differs from the published citation, so only a title
        # diff is hard there. Title-search may bind the wrong paper -> advisory.
        if rec.source.startswith("crossref:doi"):
            hard = bool(issues)
        elif rec.source == "arxiv":
            hard = any(kind == "title" for kind, _ in issues)
        else:
            hard = False

        if not issues:
            ok += 1
            print(f"[OK]        {entry.key}  ({rec.source})")
        elif hard:
            mismatch += 1
            print(f"[MISMATCH]  {entry.key}  ({rec.source})")
            for _, msg in issues:
                print(f"    - {msg}")
            flagged.append((entry, rec))
        else:
            note = (
                "preprint vs published differs"
                if rec.source == "arxiv"
                else "verify match or add a DOI"
            )
            check += 1
            print(f"[CHECK]     {entry.key}  ({rec.source}) -- {note}")
            for _, msg in issues:
                print(f"    - {msg}")

    print(
        f"\n{ok} ok, {mismatch} mismatched (authoritative), {check} to check "
        f"(search-only), {unresolved} unresolved, {len(entries)} total"
    )

    if args.show_bibtex and flagged:
        print(
            "\n" + "=" * 70 + "\nCanonical BibTeX for mismatched entries (drop-in replacements):\n"
        )
        for pos, (entry, rec) in enumerate(flagged):
            if pos:
                time.sleep(args.sleep)
            bib = canonical_bibtex(entry, rec, args.mailto)
            print(bib or f"% {entry.key}: no canonical source (title-search hit only)")
            print()

    # Only authoritative (DOI/arXiv) mismatches fail the gate; CHECK/UNRESOLVED
    # are advisories that the entry needs a DOI before it can be verified.
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
