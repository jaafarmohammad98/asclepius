"""Link subgroups to symptoms and drugs (spec A3, S2 to S6, S11).

A cluster from `asclepius.cluster.cluster_patients` only matters if it corresponds to something
clinical. `link_subgroups` tests, for every questionnaire symptom score (`score_<name>` in the
metadata) and every drug taken, whether a cluster differs from the others:

- With 2 clusters, cluster 1 against cluster 2. With more, each cluster against all the others.
- A score is compared with `asclepius.compare.compare_values`: a permutation test on ranks, Cliff's
  delta and a bootstrap interval, as for markers. Both sexes together: the clusters were already
  made within sex, and a questionnaire score has no sex-specific range.
- A drug is a yes/no value per patient (took it or not), compared the same way. On yes/no values
  Cliff's delta is the difference between the two shares, so the effect is in shares of patients.
  Patients with no drug recorded are left out of drug tests.
- Every row is re-run on the Tier A patients with the same clusters (spec S6): does the link hold
  among clinician-confirmed patients?

The rows carry no q-value or label: pass them to `run_analysis(..., subgroups=links)`, which
corrects them with every other test of the run and labels them (spec S2, S5).
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from asclepius.cluster import MIN_SHOWN, ClusterResult
from asclepius.compare import compare_values
from asclepius.dataset import LabDataset
from asclepius.run import TIER_A, TIER_A_COLUMNS, TIER_A_NAME, Settings, judge_tier_a

SUBGROUP_SCORE = "subgroup: symptom score"  # the findings table's `comparison` column
SUBGROUP_DRUG = "subgroup: drug taken"
SUBGROUP_KINDS = (SUBGROUP_SCORE, SUBGROUP_DRUG)
ALL_SEXES = "all"
SCORE_PREFIX = "score_"
DRUG_PREFIX = "drug:"
NOTHING_TO_LINK = "no stable clusters, so nothing to link"
COLUMNS = (
    "comparison",
    "marker_id",
    "marker_name",
    "sex",
    "group_a",
    "group_b",
    "subgroups_of",
    "n_a",
    "n_b",
    "median_a",
    "median_b",
    "share_a",
    "share_b",
    "effect_size",
    "ci_low",
    "ci_high",
    "p_value",
)
REFUSED_COLUMNS = ("comparison", "marker_id", "marker_name", "group_a", "group_b", "n_a", "n_b", "reason")


@dataclass(frozen=True)
class SubgroupLinks:
    """What `link_subgroups` found.

    `rows`: findings-table rows (`COLUMNS`), one per score or drug and cluster, with the Tier A
    re-run beside each (`tier_a_*`, `tier_a_result`, `holds_in_tier_a`), not yet corrected or
    labelled. `median_a`, `median_b` are score medians; `share_a`, `share_b` the share of each side
    that took the drug, hidden when fewer than 5 did or did not (spec S11). `refused`: tests not
    made, with the `reason`. `statement`: the clusters' statement, or that there is nothing to
    link. `fingerprint`, `settings`: the dataset and `Settings.record()` used, which
    `run_analysis` checks. `clusters`: the clustering's statement, reason and settings, for the
    run record (spec S7).
    """

    rows: pd.DataFrame
    refused: pd.DataFrame
    statement: str
    fingerprint: str
    settings: dict[str, object]
    clusters: dict[str, object]


def link_subgroups(
    data: LabDataset, clusters: ClusterResult, settings: Settings | None = None
) -> SubgroupLinks:
    """Test whether the clusters of `clusters` differ by symptom score and by drug taken.

    `clusters` must come from `data`. A side with fewer than `settings.min_group_size` values is
    not tested (spec S4), and nor is a drug fewer than that many clustered patients took: both go
    to `refused`. With no stable clusters there is nothing to test (spec S8). The same settings
    give the same rows.
    """
    settings = settings or Settings()
    made_from: Any = clusters.record  # plain JSON-ready values (asclepius.record.run_record)
    if made_from["dataset"]["fingerprint"] != data.fingerprint:
        raise ValueError("These clusters were made from a different dataset.")
    groups = [str(g) for g in made_from["groups"]]
    rows_out = pd.DataFrame(columns=[*COLUMNS, *TIER_A_COLUMNS, "tier_a_result", TIER_A])
    nothing = SubgroupLinks(
        rows=rows_out,
        refused=pd.DataFrame(columns=REFUSED_COLUMNS),
        statement=NOTHING_TO_LINK,
        fingerprint=data.fingerprint,
        settings=settings.record(),
        clusters={
            "statement": clusters.statement,
            "reason": clusters.reason,
            "groups": groups,
            "settings": clusters.settings,
        },
    )
    if clusters.k is None:
        return nothing

    k = clusters.k
    scores = [c for c in data.records.columns if c.startswith(SCORE_PREFIX)]
    patients = clusters.assignments.join(
        data.records.set_index("record")[["tier", "drug", *scores]], on="record"
    ).reset_index(drop=True)
    tier_a = (patients["tier"] == TIER_A_NAME).to_numpy()
    sides = [(1, "cluster 2")] if k == 2 else [(c, "other clusters") for c in range(1, k + 1)]
    subgroups_of = ", ".join(groups)

    rows: list[dict[str, object]] = []
    refused: list[dict[str, object]] = []
    for kind, marker_id, name, values in _variables(patients, scores, settings, refused):
        for cluster, other in sides:
            in_a = (patients["cluster"] == cluster).to_numpy()
            n_a, n_b = int(values[in_a].notna().sum()), int(values[~in_a].notna().sum())
            head = {
                "comparison": kind,
                "marker_id": marker_id,
                "marker_name": name,
                "group_a": f"cluster {cluster}",
                "group_b": other,
            }
            if min(n_a, n_b) < settings.min_group_size:
                refused.append(
                    head
                    | {
                        "n_a": _shown(n_a),
                        "n_b": _shown(n_b),
                        "reason": f"fewer than {settings.min_group_size} values on a side (spec S4)",
                    }
                )
                continue
            seed = [settings.seed, zlib.crc32(marker_id.encode()), cluster]
            full = _compare(values, in_a, settings, seed)
            rerun = _compare(values[tier_a], in_a[tier_a], settings, seed)
            a, b = values[in_a].dropna(), values[~in_a].dropna()
            row = head | {"sex": ALL_SEXES, "subgroups_of": subgroups_of, "n_a": n_a, "n_b": n_b}
            if kind == SUBGROUP_SCORE:
                row |= {"median_a": float(a.median()), "median_b": float(b.median())}
            else:
                row |= {"share_a": _share(a), "share_b": _share(b)}
            rows.append(row | full | {f"tier_a_{c}": v for c, v in rerun.items()})

    if rows:
        table = judge_tier_a(pd.DataFrame(rows), settings)
        for column in ("tier_a_n_a", "tier_a_n_b"):  # sizes below 5 are not shown (spec S11)
            table[column] = table[column].astype("Int64").mask(table[column] < MIN_SHOWN)
        rows_out = table.reindex(columns=rows_out.columns)
    return replace(
        nothing,
        rows=rows_out,
        refused=pd.DataFrame(refused, columns=REFUSED_COLUMNS),
        statement=clusters.statement,
    )


def _variables(
    patients: pd.DataFrame, scores: list[str], settings: Settings, refused: list[dict[str, object]]
) -> Iterator[tuple[str, str, str, pd.Series]]:
    """(kind, marker_id, name, values) for every score and every drug taken by enough patients.
    A drug's values are 1 (took it), 0 (took another) or missing (none recorded)."""
    for score in scores:
        yield SUBGROUP_SCORE, score, score.removeprefix(SCORE_PREFIX), patients[score].astype("float64")
    drug = patients["drug"].astype("string").str.strip().str.casefold().replace("", pd.NA)
    for name, takers in drug.value_counts().sort_index().items():
        marker_id = f"{DRUG_PREFIX}{name}"
        if takers < settings.min_group_size:
            refused.append(
                {
                    "comparison": SUBGROUP_DRUG,
                    "marker_id": marker_id,
                    "marker_name": name,
                    "n_a": _shown(int(takers)),
                    "reason": f"fewer than {settings.min_group_size} patients took it (spec S4)",
                }
            )
            continue
        yield SUBGROUP_DRUG, marker_id, name, (drug == name).astype("float64").where(drug.notna())


def _compare(values: pd.Series, in_a: np.ndarray, settings: Settings, seed: list[int]) -> dict[str, object]:
    """The test's columns (`n_a` to `p_value`), or only the sizes when a side has no value."""
    a, b = values[in_a].dropna(), values[~in_a].dropna()
    if a.empty or b.empty:
        return {"n_a": len(a), "n_b": len(b)}
    result = compare_values(
        a.tolist(),
        b.tolist(),
        permutations=settings.permutations,
        bootstrap=settings.bootstrap,
        rng=np.random.default_rng(seed),
    )
    return {
        "n_a": result.n_a,
        "n_b": result.n_b,
        "effect_size": result.effect_size,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "p_value": result.p_value,
    }


def _share(took: pd.Series) -> float:
    """Share of 1s, or NaN when fewer than 5 took it or fewer than 5 did not (spec S11)."""
    yes = int(took.sum())
    return float("nan") if min(yes, len(took) - yes) < MIN_SHOWN else yes / len(took)


def _shown(count: int) -> int | None:
    return count if count >= MIN_SHOWN else None
