"""Tests for factor-model output wiring in main.py.

Guards the single-source-of-truth refactor: generate_factor_models must run the
four generators in a fixed order, each with its own prefixed output paths, and
combine_factor_models must feed ap_factor_model_data those same paths in the same
order. The generators and ap_factor_model_data are mocked so the expensive data
pipeline never runs.
"""

from __future__ import annotations

from pathlib import Path

from jkp.data import main

PREFIXES = ["ff", "hxz", "mp", "dhs"]
GENERATOR_NAMES = {
    "ff": "gen_ff_data",
    "hxz": "gen_hxz_data",
    "mp": "gen_mispricing_data",
    "dhs": "gen_dhs_data",
}


def test_factor_paths_derives_from_prefix(tmp_path):
    assert main.factor_paths(tmp_path, "ff") == (
        tmp_path / "ff_factors_monthly.parquet",
        tmp_path / "ff_factors_daily.parquet",
        tmp_path / "ff_characteristics.parquet",
    )


def test_generate_factor_models_runs_generators_in_order(monkeypatch, tmp_path):
    calls: list[tuple[str, Path, Path, Path]] = []

    def make_generator(prefix: str):
        def generator(paths, monthly, daily, characteristics):
            calls.append((prefix, monthly, daily, characteristics))

        return generator

    for prefix, name in GENERATOR_NAMES.items():
        monkeypatch.setattr(main, name, make_generator(prefix))

    outputs = main.generate_factor_models(paths=object(), interim=tmp_path)

    # Generators run in order, each with its own prefixed paths.
    assert [c[0] for c in calls] == PREFIXES
    for prefix, monthly, daily, characteristics in calls:
        assert (monthly, daily, characteristics) == main.factor_paths(tmp_path, prefix)

    # Returned mapping matches what each generator was handed.
    assert list(outputs) == PREFIXES
    assert outputs == {prefix: main.factor_paths(tmp_path, prefix) for prefix in PREFIXES}


def test_combine_factor_models_feeds_ap_in_order(monkeypatch, tmp_path):
    ap_kwargs: dict = {}
    monkeypatch.setattr(main, "ap_factor_model_data", lambda **kw: ap_kwargs.update(kw))

    factor_outputs = {prefix: main.factor_paths(tmp_path, prefix) for prefix in PREFIXES}
    main.combine_factor_models(tmp_path, factor_outputs)

    assert ap_kwargs["monthly_factor_inputs"] == [factor_outputs[p][0] for p in PREFIXES]
    assert ap_kwargs["daily_factor_inputs"] == [factor_outputs[p][1] for p in PREFIXES]
    assert ap_kwargs["chars_inputs"] == [factor_outputs[p][2] for p in PREFIXES]
