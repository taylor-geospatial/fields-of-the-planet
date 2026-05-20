"""Validate every entry in paper/refs.bib against the Semantic Scholar API.

For each bib entry we extract the title, query the S2 ``graph/v1/paper/search``
endpoint, and report the canonical venue/year/authors. If our entry currently
points at an arXiv preprint but S2 lists a published conference/journal
version, we surface that as a suggested swap.

Output: paper/scripts/output/refs_audit.csv  (one row per bib entry).

We do not auto-edit refs.bib; the operator reviews the CSV and applies the
swaps they agree with.
"""

import csv
import re
import time
from pathlib import Path

import requests

BIB = Path("paper/refs.bib")
OUT = Path("paper/scripts/output/refs_audit.csv")
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,venue,authors,externalIds,publicationVenue,publicationTypes,journal"


def parse_bib(path: Path):
    text = path.read_text()
    # Split on entry markers; keep the entry leader.
    entries = re.split(r"\n(?=@\w+\{)", text.strip())
    out = []
    for e in entries:
        m = re.match(r"@(\w+)\{([^,]+),", e)
        if not m:
            continue
        kind, key = m.group(1), m.group(2)
        title = _field(e, "title")
        year = _field(e, "year")
        venue = _field(e, "booktitle") or _field(e, "journal")
        is_arxiv = bool(re.search(r"arxiv|arXiv", e))
        out.append({
            "key": key, "kind": kind, "title": title, "year": year,
            "current_venue": venue, "looks_arxiv": is_arxiv,
            "raw": e,
        })
    return out


def _field(entry: str, name: str) -> str | None:
    """Read a single bib field; strips outer braces, collapses whitespace."""
    m = re.search(rf"{name}\s*=\s*\{{", entry)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    for i in range(start, len(entry)):
        c = entry[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                inner = entry[start + 1:i]
                # Strip nested braces and collapse whitespace.
                inner = re.sub(r"[{}]", "", inner)
                inner = re.sub(r"\s+", " ", inner).strip()
                return inner
    return None


def search_s2(title: str) -> dict | None:
    if not title:
        return None
    params = {"query": title, "limit": 3, "fields": FIELDS}
    for attempt in range(3):
        try:
            r = requests.get(S2_URL, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as err:
            print(f"  S2 error for '{title[:60]}...': {err}")
            return None
        for hit in data.get("data", []):
            # Loose title match: lowercased, alphanumerics only.
            if _norm(hit["title"]) == _norm(title):
                return hit
        if data.get("data"):
            return data["data"][0]
        return None
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def classify(s2_hit: dict | None) -> tuple[str, str, str]:
    if not s2_hit:
        return "no_match", "", ""
    venue = s2_hit.get("venue") or ""
    pub_venue = (s2_hit.get("publicationVenue") or {}).get("name") or ""
    journal = (s2_hit.get("journal") or {}).get("name") or ""
    types = s2_hit.get("publicationTypes") or []
    external = s2_hit.get("externalIds") or {}
    is_arxiv_only = "ArXiv" in external and not any(
        external.get(k) for k in ("DOI", "DBLP", "MAG")
    )
    final_venue = pub_venue or venue or journal
    kind = "preprint" if is_arxiv_only else "venue"
    return kind, final_venue, ",".join(types)


def main():
    entries = parse_bib(BIB)
    OUT.parent.mkdir(exist_ok=True, parents=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "key", "current_kind", "current_year", "current_venue",
            "currently_arxiv", "s2_match_title", "s2_year", "s2_venue",
            "s2_pub_types", "swap_suggested",
        ])
        for e in entries:
            print(f"checking {e['key']}...")
            hit = search_s2(e["title"])
            if not hit:
                w.writerow([e["key"], e["kind"], e["year"], e["current_venue"],
                            e["looks_arxiv"], "", "", "", "", "no_match"])
                continue
            kind, s2_venue, s2_types = classify(hit)
            swap = ""
            if e["looks_arxiv"] and kind == "venue":
                swap = f"swap_to:{s2_venue}"
            elif not e["looks_arxiv"] and kind == "preprint":
                swap = "downgrade_to_arxiv"
            elif kind == "venue" and s2_venue and s2_venue.lower() not in (e["current_venue"] or "").lower():
                swap = f"verify_venue:{s2_venue}"
            w.writerow([
                e["key"], e["kind"], e["year"], e["current_venue"],
                e["looks_arxiv"],
                hit.get("title", ""), hit.get("year", ""), s2_venue, s2_types,
                swap,
            ])
            time.sleep(1.1)  # S2 unauthenticated rate limit
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
