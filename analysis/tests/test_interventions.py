import json
from pathlib import Path

import pandas as pd
import pytest

from asclepius import synthetic
from asclepius.interventions import CLEAR, rank_interventions
from asclepius.record import export_run
from asclepius.run import Settings
from asclepius.selfreport import load_self_reports


def self_reports(tmp_path, *, seed=1, authors=1000, outcomes=(), null=False):
    out = synthetic.write_self_reports(
        tmp_path, authors=authors, outcomes=list(outcomes), null=null, seed=seed
    )
    return load_self_reports(out.export)


def row(result, intervention):
    rows = result.findings[result.findings["marker_id"] == f"intervention:{intervention}"]
    assert len(rows) == 1
    return rows.iloc[0]


PLANTED = [
    synthetic.PlantedOutcome("tadalafil", improved=0.6, worse=0.05),
    synthetic.PlantedOutcome("cyproheptadine", improved=0.1, worse=0.45),
]


def test_planted_outcome_shares_are_recovered_within_their_intervals(tmp_path):
    # a 95% interval misses about 1 run in 20, so count over 20 seeds
    inside, runs = 0, 0
    for seed in range(20):
        result = rank_interventions(self_reports(tmp_path / str(seed), seed=seed, outcomes=PLANTED))
        for p in PLANTED:
            r = row(result, p.intervention)
            assert r["n_clear"] >= 10
            inside += r["improved_low"] <= p.improved <= r["improved_high"]
            inside += r["worse_low"] <= p.worse <= r["worse_high"]
            runs += 2
    assert inside / runs >= 0.85


def test_the_most_improved_ranks_first_and_the_most_worsened_is_a_safety_signal(tmp_path):
    result = rank_interventions(self_reports(tmp_path, outcomes=PLANTED))
    assert row(result, "tadalafil")["rank"] == 1
    assert list(result.findings["rank"].dropna()) == list(range(1, result.findings["rank"].notna().sum() + 1))
    lows = result.findings.dropna(subset=["rank"])["improved_low"]
    assert lows.is_monotonic_decreasing
    assert "cyproheptadine" in set(result.safety["marker_name"])
    assert "tadalafil" not in set(result.safety["marker_name"])
    assert (result.safety["worse_low"] >= 0.2).all()


def with_extra_reports(tmp_path, rows, *, authors=300, outcomes=()):
    """A synthetic export with `rows` of (author index, intervention, outcome) added, each on that
    author's first post; authors are numbered by their first post's order."""
    out = synthetic.write_self_reports(tmp_path, authors=authors, outcomes=list(outcomes), seed=3)
    cases = pd.read_csv(out.export / "cases.csv", dtype=str)
    firsts = cases[cases["kind"] == "submission"]["case_id"].tolist()
    path = out.export / "interventions.csv"
    extra = pd.DataFrame(
        [
            {"case_id": firsts[i], "intervention": name, "outcome": outcome, "duration_weeks": ""}
            for i, name, outcome in rows
        ]
    )
    pd.concat([pd.read_csv(path, dtype=str), extra]).to_csv(path, index=False)
    return out.export, cases


def test_an_intervention_reported_by_9_authors_is_refused_and_by_10_is_ranked(tmp_path):
    rows = [(i, "acupuncture", "partial_improvement") for i in range(9)]
    rows += [(i, "sauna", "no_effect") for i in range(10)]
    export, _ = with_extra_reports(tmp_path, rows)
    result = rank_interventions(load_self_reports(export))
    refused = result.refused[result.refused["marker_name"] == "acupuncture"]
    assert len(refused) == 1 and "fewer than 10" in refused.iloc[0]["reason"]
    assert refused.iloc[0]["n_clear"] == 9
    assert "acupuncture" not in set(result.findings["marker_name"])
    sauna = row(result, "sauna")
    assert sauna["n_clear"] == 10 and sauna["no_effect"] == 10 and sauna["share_improved"] == 0


def test_unclear_outcomes_are_left_out_of_the_shares_and_counted(tmp_path):
    rows = [(i, "sauna", "partial_improvement") for i in range(10)]
    rows += [(i, "sauna", "worse") for i in range(10, 15)]
    rows += [(i, "sauna", "unclear") for i in range(15, 21)]
    export, _ = with_extra_reports(tmp_path, rows)
    sauna = row(rank_interventions(load_self_reports(export)), "sauna")
    assert (sauna["n_clear"], sauna["unclear"], sauna["n_authors"]) == (15, 6, 21)
    assert sauna["share_improved"] == pytest.approx(10 / 15)
    assert sauna["share_worse"] == pytest.approx(5 / 15)


def test_an_author_who_reported_an_intervention_twice_counts_once_with_the_latest_outcome(tmp_path):
    out = synthetic.write_self_reports(tmp_path, authors=300, seed=3)
    cases = pd.read_csv(out.export / "cases.csv", dtype=str)
    posts = cases[cases["author_key"].notna()].groupby("author_key")
    repeat = next(g.sort_values("year") for _, g in posts if g["year"].nunique() >= 2)
    older, newer = repeat["case_id"].iloc[0], repeat["case_id"].iloc[-1]
    firsts = [c for c in cases[cases["kind"] == "submission"]["case_id"] if c not in set(repeat["case_id"])]
    extra = [(c, "sauna", "partial_improvement") for c in firsts[:10]]
    extra += [(older, "sauna", "worse"), (newer, "sauna", "major_improvement")]
    path = out.export / "interventions.csv"
    table = pd.DataFrame(
        [{"case_id": c, "intervention": n, "outcome": o, "duration_weeks": ""} for c, n, o in extra]
    )
    pd.concat([pd.read_csv(path, dtype=str), table]).to_csv(path, index=False)
    sauna = row(rank_interventions(load_self_reports(out.export)), "sauna")
    assert sauna["n_clear"] == 11  # 10 authors and the one who posted twice, once
    assert sauna["worse"] == 0  # their older "worse" was replaced by the newer outcome


def ok(count):
    return count == 0 or count >= 5


def assert_no_count_below_5(result):
    """No count from 1 to 4 is shown, or can be worked out from the shares and other counts."""
    for table in (result.findings, result.safety, result.refused):
        assert not {"author", "author_key", "case_id"} & set(table.columns)
        for column in [c for c in table.columns if c.startswith("n_") or c in CLEAR or c == "unclear"]:
            assert all(ok(int(n)) for n in table[column].dropna()), column
    for _, r in result.findings.iterrows():
        n = int(r["n_clear"])
        known = {}
        if pd.notna(r["share_improved"]):
            known["improved"] = round(r["share_improved"] * n)
            assert ok(known["improved"]) and ok(n - known["improved"])
        if pd.notna(r["share_worse"]):
            known["worse"] = round(r["share_worse"] * n)
            assert ok(known["worse"]) and ok(n - known["worse"])
        if len(known) == 2:
            assert ok(n - known["improved"] - known["worse"])
        if "improved" in known:  # one of major and partial shown gives the other back
            parts = [r["major_improvement"], r["partial_improvement"]]
            if sum(pd.notna(p) for p in parts) == 1:
                assert ok(known["improved"] - int(next(p for p in parts if pd.notna(p))))
        if pd.notna(r["n_authors"]):
            assert ok(int(r["n_authors"]) - n)


@pytest.mark.parametrize("seed", range(5))
def test_no_count_below_5_is_shown_or_given_back(tmp_path, seed):
    result = rank_interventions(self_reports(tmp_path, seed=seed, authors=150, outcomes=PLANTED))
    assert len(result.findings) and result.findings["share_improved"].isna().any()  # some hidden here
    assert result.findings.loc[result.findings["share_improved"].isna(), "rank"].isna().all()
    assert_no_count_below_5(result)


ADVICE = ("recommend", "effective", "efficacy", "should try", "works for", "best treatment")


def all_text(result):
    yield result.statement
    yield str(result.record.get("note"))
    for table in (result.findings, result.safety, result.refused):
        yield from table.columns
        for column in table.columns:
            yield from (str(v) for v in table[column] if isinstance(v, str))


def test_every_table_says_reported_outcome_and_nothing_reads_as_advice(tmp_path):
    rows = [(i, "acupuncture", "partial_improvement") for i in range(9)]  # refused
    export, _ = with_extra_reports(tmp_path, rows, authors=1000, outcomes=PLANTED)
    result = rank_interventions(load_self_reports(export))
    for table in (result.findings, result.safety, result.refused):
        assert len(table)
        assert table["caveat"].str.contains("reported outcome, not evidence that it works").all()
        assert table["caveat"].str.contains("self-report, no controls").all()
    assert "reported outcome, not evidence that it works" in result.statement
    for text in all_text(result):
        assert not any(word in text.lower() for word in ADVICE), text


def test_settings_are_recorded_the_run_is_repeatable_and_it_exports(tmp_path):
    data = self_reports(tmp_path / "sr", outcomes=PLANTED)
    a, b = rank_interventions(data), rank_interventions(data)
    assert a.findings.equals(b.findings) and a.safety.equals(b.safety)
    assert a.record["analysis"] == "intervention outcomes"
    assert a.record["settings"] == Settings().record() | {"worse_signal": 0.2}
    out = export_run(a, tmp_path / "out")
    record = json.loads((out / "run_record.json").read_text())
    assert set(record["outputs"]) == {"findings.csv", "refused.csv", "skipped.csv", "safety.csv"}
    assert "reported outcome, not evidence that it works" in (out / "safety.csv").read_text()


REAL_EXPORT = Path(__file__).parents[2] / "harvester" / "data" / "export"


@pytest.mark.skipif(not (REAL_EXPORT / "cases.csv").exists(), reason="the r/PSSD export is not here")
def test_the_real_export_gives_a_ranked_table_with_its_labels_and_no_small_count():
    result = rank_interventions(load_self_reports(REAL_EXPORT))
    assert result.findings["rank"].notna().sum() >= 20
    assert (result.findings["caveat"].str.contains("not evidence that it works")).all()
    assert_no_count_below_5(result)
    for text in all_text(result):
        assert not any(word in text.lower() for word in ADVICE), text


def test_a_high_share_worse_is_flagged_even_when_its_share_must_be_hidden(tmp_path):
    # 2 authors not worse: the share worse would give that 2 back, so it is hidden, but still flagged
    rows = [(i, "sauna", "worse") for i in range(12)] + [
        (i, "sauna", "partial_improvement") for i in range(12, 14)
    ]
    export, _ = with_extra_reports(tmp_path, rows)
    result = rank_interventions(load_self_reports(export))
    sauna = row(result, "sauna")
    assert pd.isna(sauna["share_worse"]) and sauna["safety_signal"]
    assert "sauna" in set(result.safety["marker_name"])


def test_an_unranked_row_says_why(tmp_path):
    rows = [(i, "sauna", "major_improvement") for i in range(12)] + [(12, "sauna", "no_effect")]
    export, _ = with_extra_reports(tmp_path, rows)
    sauna = row(rank_interventions(load_self_reports(export)), "sauna")
    assert pd.isna(sauna["rank"]) and "fewer than 5" in sauna["not_ranked"]
    ranked = rank_interventions(load_self_reports(export)).findings.dropna(subset=["rank"])
    assert ranked["not_ranked"].isna().all()


def test_the_refused_list_names_no_intervention_fewer_than_5_authors_reported(tmp_path):
    rows = [(i, "acupuncture", "partial_improvement") for i in range(7)] + [(0, "hypnosis", "worse")]
    export, _ = with_extra_reports(tmp_path, rows)
    refused = rank_interventions(load_self_reports(export)).refused
    assert "acupuncture" in set(refused["marker_name"])
    assert "hypnosis" not in set(refused["marker_name"])
    rare = refused[refused["marker_id"] == "intervention:fewer than 5 authors"]
    assert len(rare) == 1 and "not named" in rare.iloc[0]["reason"]
