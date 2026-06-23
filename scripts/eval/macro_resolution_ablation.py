"""Macro-average the 3-condition resolution ablation across the dense-10.

Reads per-country `<country>.csv.bins.csv` under logs/resolution_ablation/<cond>/
for cond in {planet3m, s2up, s2nat10}, and macro-averages PQ / RQ.5 / AP per area
bin across countries (each country weighted equally). Prints a table and writes
logs/resolution_ablation_macro.csv.
"""

from pathlib import Path

ROOT = Path("logs/resolution_ablation")
CONDS = ["planet3m", "s2up", "s2nat10"]
BINS = ["small", "medium", "large", "all"]


def _read_bins(p):
    """bin -> (pq, rq_50, ap) from a .bins.csv (area_bins field has an embedded comma)."""
    out = {}
    for line in p.read_text().splitlines()[1:]:
        f = line.split(",")
        if len(f) < 10:
            continue
        # f: 0.5 | 2 | bin | n_gt | n_pred | pq | sq | rq_50 | f1_75 | ap_5_95
        out[f[2]] = (float(f[5]), float(f[7]), float(f[9]))
    return out


def main() -> int:
    rows = []
    macro = {c: {b: {"pq": [], "rq": [], "ap": []} for b in BINS} for c in CONDS}
    for cond in CONDS:
        d = ROOT / cond
        for bp in sorted(d.glob("*.csv.bins.csv")):
            for b, (pq, rq, ap) in _read_bins(bp).items():
                if b in macro[cond]:
                    macro[cond][b]["pq"].append(pq)
                    macro[cond][b]["rq"].append(rq)
                    macro[cond][b]["ap"].append(ap)

    def avg(x):
        return 100 * sum(x) / len(x) if x else float("nan")

    print(f"{'bin':7s} | " + " | ".join(f"{c:>22s}" for c in CONDS))
    print(f"{'':7s} | " + " | ".join(f"{'PQ':>6} {'RQ.5':>6} {'AP':>6}" for _ in CONDS))
    for b in BINS:
        cells = []
        for c in CONDS:
            m = macro[c][b]
            cells.append(f"{avg(m['pq']):6.1f} {avg(m['rq']):6.1f} {avg(m['ap']):6.1f}")
            rows.append((c, b, len(m["pq"]), avg(m["pq"]), avg(m["rq"]), avg(m["ap"])))
        print(f"{b:7s} | " + " | ".join(cells))

    out = Path("logs/resolution_ablation_macro.csv")
    with out.open("w") as f:
        f.write("condition,bin,n_countries,PQ,RQ_50,AP\n")
        for cond, b, n, pq, rq, ap in rows:
            f.write(f"{cond},{b},{n},{pq:.2f},{rq:.2f},{ap:.2f}\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
