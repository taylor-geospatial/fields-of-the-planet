"""Regenerate every script-generated LaTeX table in the paper from its source CSV.

One command rebuilds all ``\\input{figs/*.tex}`` tables so prose/tables cannot
drift from the canonical eval CSVs in ``logs/``. Handwritten tables
(tab:scope, tab:udm2, tab:upsampled_s2[_main]) are not produced here; their
sources are documented in ``PROVENANCE.md``.

Run from the repo root::

    uv run --no-sync python paper/scripts/build_tables.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (generator script, output .tex it writes). Order is independent -- each reads
# its own source CSV(s).
GENERATORS = [
    ("polygon_metrics_table.py", "polygon_metrics.tex"),
    ("area_bins_table.py", "area_bins.tex"),
    ("heldout_results_table.py", "heldout_results.tex"),
    ("heldout_per_country_table.py", "heldout_per_country.tex"),
    ("full_data_table.py", "full_data_compare.tex"),
]


def main() -> int:
    failures: list[str] = []
    for script, out in GENERATORS:
        print(f"=== {script} -> figs/{out} ===")
        result = subprocess.run([sys.executable, str(HERE / script)], check=False)
        if result.returncode != 0:
            failures.append(script)
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(GENERATORS)} tables regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
