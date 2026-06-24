"""Command-line entry point for training and evaluating FTP models.

ftw-planet check-data
ftw-planet train <config>
ftw-planet eval --ckpt <checkpoint>
ftw-planet reproduce --ckpt <checkpoint>
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ftw_planet.evaluation import (
    DENSE_LABEL_COUNTRIES,
    FULLDATA_REGIONS,
    macro_average,
)

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "scripts" / "eval"
SPLITS = {"dense10": DENSE_LABEL_COUNTRIES, "full23": FULLDATA_REGIONS}

# Published dense-label held-out numbers for the released PRUE-FTP-B3 checkpoint.
REFERENCE = {"pq": 0.355, "object_ws_f1": 0.452, "pixel_level_iou": 0.688}


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def check_data(countries: tuple[str, ...]) -> list[str]:
    planet = REPO / "data" / "planet"
    return [c for c in countries if not (planet / c).is_dir()]


def cmd_check_data(args: argparse.Namespace) -> int:
    countries = SPLITS[args.split]
    missing = check_data(countries)
    if missing:
        print(f"missing data/planet/<country> for: {', '.join(missing)}")
        print("Expected layout: data/planet/<country>/window_*/<patch>.tif (+ labels).")
        print("See docs/DATASET.md for how to obtain the dataset.")
        return 1
    print(f"data present for all {len(countries)} {args.split} countries.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cmd = ["ftw", "model", "fit", "-c", args.config]
    if args.resume:
        cmd += ["--ckpt_path", args.resume]
    if args.extra:
        cmd += ["--", *args.extra]
    run(cmd)
    return 0


def evaluate(ckpt: Path, split: str, watershed: bool, tta: bool, out_dir: Path) -> dict[str, float]:
    countries = SPLITS[split]
    out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--ckpt",
        str(ckpt),
        "--countries",
        *countries,
        "--min-pad-size",
        "512",
        "--dataset-backend",
        "planet",
    ]
    if watershed:
        common.append("--watershed")
    if tta:
        common.append("--tta")

    postprocess_csv = out_dir / "postprocess.csv"
    polygon_csv = out_dir / "polygon_metrics.csv"
    run(
        [
            sys.executable,
            str(EVAL_DIR / "postprocess_eval.py"),
            "--out",
            str(postprocess_csv),
            *common,
        ]
    )
    run(
        [
            sys.executable,
            str(EVAL_DIR / "polygon_metrics_eval.py"),
            "--out",
            str(polygon_csv),
            *common,
        ]
    )

    summary = macro_average(postprocess_csv, countries)
    summary.update(macro_average(polygon_csv, countries))
    return summary


def print_summary(split: str, summary: dict[str, float]) -> None:
    print(
        f"\nMacro-average over {summary['n_countries']}/{summary['n_expected']} {split} countries:"
    )
    rows = [
        ("PQ", "pq"),
        ("SQ", "pq_sq"),
        ("RQ", "pq_rq"),
        ("F1[.5:.95]", "ap_5_95"),
        ("Obj F1 (WS+TTA)", "object_ws_f1"),
        ("Pixel IoU", "pixel_level_iou"),
        ("|delta N_poly|", "polygon_count_delta_mean"),
        ("Boundary err (m)", "boundary_error_m_mean"),
    ]
    for label, key in rows:
        if key not in summary:
            continue
        ref = REFERENCE.get(key)
        suffix = f"   (paper {ref:.3f})" if ref is not None and split == "dense10" else ""
        print(f"  {label:18s} {summary[key]:.4f}{suffix}")


def cmd_eval(args: argparse.Namespace) -> int:
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}")
        return 1
    out_dir = Path(args.out) if args.out else REPO / "logs" / "eval" / ckpt.stem
    summary = evaluate(ckpt, args.split, not args.no_ws, not args.no_tta, out_dir)
    print_summary(args.split, summary)
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    if args.train:
        cmd_train(argparse.Namespace(config=args.train, resume=None, extra=[]))
    return cmd_eval(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftw-planet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-data", help="validate the local dataset layout")
    p.add_argument("--split", choices=SPLITS, default="dense10")
    p.set_defaults(func=cmd_check_data)

    p = sub.add_parser("train", help="train a model from a config")
    p.add_argument("config")
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="passed through to ftw model fit")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=SPLITS, default="dense10")
    p.add_argument("--no-ws", action="store_true", help="disable watershed post-processing")
    p.add_argument("--no-tta", action="store_true", help="disable test-time augmentation")
    p.add_argument("--out", help="output directory for CSVs")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser(
        "reproduce", help="evaluate (optionally after training) and print the headline table"
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=SPLITS, default="dense10")
    p.add_argument("--train", help="config to train first, then evaluate the resulting checkpoint")
    p.add_argument("--no-ws", action="store_true")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_reproduce)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
