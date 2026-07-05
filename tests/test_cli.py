import textwrap

import pytest

from ftw_planet import cli, evaluation


def test_dense_label_excludes_kenya():
    assert "kenya" not in evaluation.DENSE_LABEL_COUNTRIES
    assert "kenya" in evaluation.HELDOUT_COUNTRIES
    assert len(evaluation.DENSE_LABEL_COUNTRIES) == 10


def test_macro_average(tmp_path):
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            country,pixel_level_iou,object_ws_f1
            belgium,0.70,0.60
            germany,0.80,0.40
            kenya,0.00,0.00
            """
        )
    )
    summary = evaluation.macro_average(
        csv, evaluation.DENSE_LABEL_COUNTRIES, metrics=("pixel_level_iou", "object_ws_f1")
    )
    assert summary["n_countries"] == 2
    assert summary["pixel_level_iou"] == pytest.approx(0.75)
    assert summary["object_ws_f1"] == pytest.approx(0.50)


def test_check_data_reports_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO", tmp_path)
    (tmp_path / "data" / "planet" / "belgium").mkdir(parents=True)
    missing = cli.check_data(("belgium", "germany"))
    assert missing == ["germany"]


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_parser_eval_needs_ckpt():
    args = cli.build_parser().parse_args(["eval", "--ckpt", "x.ckpt"])
    assert args.split == "dense10"
    assert args.ckpt == "x.ckpt"
