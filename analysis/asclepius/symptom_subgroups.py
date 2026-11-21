"""Symptom subgroups in self-reports (spec A5, S4, S8 to S11).

The lab-value clustering (`asclepius.cluster`) waits for real reports, but symptom profiles from
thousands of authors can already show whether PSSD splits into subtypes. `cluster_symptoms`
clusters authors by which symptoms they report, then tests whether the clusters differ by causal
drug class, onset or trajectory:

1. One row per author (`asclepius.profile.author_profiles`, spec S9), one yes/no feature per
   symptom at least `min_reporters` authors report. The catch-all "other" is not a feature: it
   says nothing about which symptom. Authors with fewer than `min_symptoms` of the features are
   left out, and the result says how many.
2. The distance between two authors is the Jaccard distance: of the symptoms either reports, the
   share only one of them does. It suits yes/no data, where not reporting a symptom says less than
   reporting it: two authors are not alike for both leaving out the same 40 symptoms.
3. k-medoids (k-means++-style starts, best of several) on that distance splits the authors into 2
   to `k_max` clusters, and the number of clusters is chosen by the same subsample stability as
   the lab clustering (`asclepius.cluster.choose_k`, adjusted Rand index). Every cluster must have
   at least `min_cluster_size` authors.
4. Stability alone is not enough, so the clustering must also be tighter than in the same data
   with the links between symptoms broken: a swap randomisation (curveball, Strona et al. 2014)
   that keeps how many symptoms each author reports and how many authors report each symptom. It
   is a permutation test (spec S1), and it rules out clusters that only split authors who report
   many symptoms from those who report few. Failing either test gives "no stable clusters" (S8).
5. Each cluster is described by its most over-represented symptoms: the share of the cluster that
   reports it against the share of the other authors.
6. Clusters are compared by drug class, the most common single causal drugs, onset during use and
   each trajectory with the drug
   profile's Fisher test, odds ratio and intervals (`asclepius.profile.compare_shares`); every row
   is corrected together and labelled by `asclepius.run.label_findings` with `self_report=True`
   (spec S2, S5, S10). Symptoms are not tested: the clusters were made from them.

No count below 5 is shown (spec S11). Thresholds were chosen without a clinician and are marked
provisional until M3-11.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from asclepius.cluster import MIN_SHOWN, NO_STABLE, TooFewPatients, choose_k, count_text, number_by_size
from asclepius.profile import (
    CLASS,
    CLASSES,
    COLUMNS,
    DRUG,
    NOTE,
    ONSET_AFTER,
    ONSET_DURING,
    PROFILE_COLUMNS,
    REFUSED_COLUMNS,
    TOP_DRUGS,
    TRAJECTORIES,
    author_profiles,
    compare_shares,
    hide_small_cells,
    hide_together,
    single_drugs,
)
from asclepius.ratios import STATUS
from asclepius.record import run_record
from asclepius.run import NOTHING_RELIABLE, RunResult, Settings, label_findings, statement
from asclepius.selfreport import SelfReportDataset

ANALYSIS = "symptom subgroups"  # the run record's `analysis`
CATCH_ALL = frozenset({"other"})  # symptoms that name no symptom: not features
SUBGROUP_CLASS = "symptom subgroup: drug class"  # the findings table's `comparison` column
SUBGROUP_DRUG = "symptom subgroup: causal drug"
SUBGROUP_ONSET = "symptom subgroup: onset"
SUBGROUP_TRAJECTORY = "symptom subgroup: trajectory"
LEVEL = "cluster"  # the `level` column of onset and trajectory rows; class and drug rows have their own
SUMMARY_COLUMNS = ("cluster", "n_authors", "symptom", "symptom_name", "share_in", "share_out", "difference")


@dataclass(frozen=True)
class SymptomClusterSettings:
    """Every threshold `cluster_symptoms` uses; `record()` is what a run stores.

    Clustering is refused below `min_authors` authors with at least `min_symptoms` symptoms (spec
    S4). A symptom is a feature when at least `min_reporters` authors report it. A clustering is
    accepted when its stability (mean adjusted Rand index over `subsamples` subsamples of
    `subsample_share` of the authors) is at least `min_stability`, every cluster has at least
    `min_cluster_size` authors, and it is tighter than in all but `null_alpha` of `null_samples`
    swap-randomised datasets (`swap_rounds` rounds of pairing every author with another). Each
    cluster is described by its `top_symptoms` most over-represented symptoms, and compared by the
    `top_drugs` most common single drugs as well as by drug class.
    """

    min_authors: int = 30
    min_symptoms: int = 2
    min_reporters: int = 10
    k_max: int = 6
    min_cluster_size: int = 30
    min_stability: float = 0.8
    subsamples: int = 20
    subsample_share: float = 0.8
    null_samples: int = 49
    null_alpha: float = 0.05
    restarts: int = 4
    swap_rounds: int = 20
    top_symptoms: int = 5
    top_drugs: int = TOP_DRUGS
    seed: int = 0
    distance: str = "Jaccard"
    one_group_check: str = field(
        default="swap randomisation keeping each author's and each symptom's count: a permutation test"
    )
    status: str = field(default=STATUS)

    def __post_init__(self) -> None:
        if self.min_authors < 2 or self.min_cluster_size < 1 or self.min_reporters < 1:
            raise ValueError(
                "min_authors must be at least 2, and min_cluster_size and min_reporters at least 1."
            )
        if self.min_symptoms < 1:
            raise ValueError("min_symptoms must be at least 1.")
        if self.top_drugs < 0:
            raise ValueError("top_drugs cannot be negative.")
        if self.k_max < 2:
            raise ValueError("k_max must be at least 2.")
        if not 0 < self.subsample_share < 1:
            raise ValueError("subsample_share must be between 0 and 1.")
        if not 0 <= self.min_stability <= 1 or not 0 < self.null_alpha <= 1:
            raise ValueError("min_stability must be between 0 and 1, and null_alpha above 0 and at most 1.")
        if min(self.subsamples, self.null_samples, self.restarts, self.swap_rounds, self.top_symptoms) < 1:
            raise ValueError(
                "subsamples, null_samples, restarts, swap_rounds and top_symptoms must be at least 1."
            )

    def record(self) -> dict[str, object]:
        return asdict(self)


class TooFewAuthors(TooFewPatients):
    """Clustering was refused: fewer authors than `SymptomClusterSettings.min_authors` (spec S4).
    A `TooFewPatients`, so a caller catching the lab refusal catches this one too."""


@dataclass(frozen=True)
class SymptomSubgroups(RunResult):
    """What `cluster_symptoms` found.

    `clusters`: "2 stable clusters" (or 3, ...) or "no stable clusters", with `reason` saying why;
    `k` the number, None when there are none. `findings`: each cluster against the other(s) by drug
    class, single causal drug, onset and trajectory (`asclepius.profile.COLUMNS`), with `q_value`, `label` and `caveat`;
    `refused`: those not tested, with the `reason`. `summary`: each cluster's `top_symptoms` most
    over-represented symptoms (`SUMMARY_COLUMNS`). `sizes`: authors per cluster. `stability`: per k,
    as in `asclepius.cluster.choose_k`. `null_p`: the swap test's p-value. `symptoms`: the features
    used. `left_out`: authors with fewer than `min_symptoms` of them. `statement`: the clusters'
    statement, and the findings' (spec S8). Counts below 5 are None (S11). `settings`, `record`: as
    for `run_analysis`, with the clustering settings under `clusters` (spec S7).
    """

    clusters: str
    reason: str
    k: int | None
    summary: pd.DataFrame
    sizes: pd.DataFrame
    stability: pd.DataFrame
    null_p: float | None
    symptoms: tuple[str, ...]
    left_out: int | None


def cluster_symptoms(
    data: SelfReportDataset,
    settings: Settings | None = None,
    cluster_settings: SymptomClusterSettings | None = None,
) -> SymptomSubgroups:
    """Cluster authors by the symptoms they report and test whether the clusters differ by drug
    class, onset and trajectory (spec A5). Refuses, with `TooFewAuthors`, when fewer than
    `cluster_settings.min_authors` authors report enough symptoms (spec S4). The same data and
    settings give the same result."""
    settings = settings or Settings()
    cs = cluster_settings or SymptomClusterSettings()
    profiles = author_profiles(data)
    reported = profiles.drop(columns=list(PROFILE_COLUMNS)).astype(bool)
    reported = reported.drop(columns=[c for c in reported.columns if c in CATCH_ALL])
    symptoms = tuple(c for c in reported.columns if reported[c].sum() >= cs.min_reporters)
    enough = reported[list(symptoms)].sum(axis=1) >= cs.min_symptoms
    left_out = int((~enough).sum())
    authors = profiles[enough]
    if len(authors) < cs.min_authors:
        raise TooFewAuthors(
            f"Refused: {count_text(len(authors))} authors report at least {cs.min_symptoms} symptoms "
            f"({count_text(left_out)} more report fewer), and clustering needs at least {cs.min_authors} "
            f"(spec S4, {cs.status})."
        )
    x = reported.loc[enough, list(symptoms)].to_numpy()
    d = jaccard_distances(x)
    rng = np.random.default_rng(cs.seed)
    recorded = settings.record() | {"clusters": cs.record()}
    base: dict = {
        "symptoms": symptoms,
        "left_out": _shown(left_out),
        "skipped": pd.DataFrame(columns=["level", "group", "reason"]),
        "settings": recorded,
    }

    choice = choose_k(
        len(x), lambda rows, k: kmedoids(d[np.ix_(rows, rows)], k, cs.restarts, rng), cs, rng, "authors"
    )
    base["stability"] = choice.stability
    if choice.k is None:
        return _no_clusters(data, choice.reason, None, **base)
    k = choice.k
    null_p = _null_p(x, k, spread(d, choice.fits[k]), cs, rng)
    if null_p > cs.null_alpha:
        reason = (
            f"{k} clusters are stable but no tighter than with the links between symptoms broken "
            f"(p = {null_p:.2f})"
        )
        return _no_clusters(data, reason, null_p, **base)

    cluster = pd.Series(number_by_size(choice.fits[k], k), index=authors.index)
    sizes = cluster.value_counts().sort_index().rename_axis("cluster").rename("n_authors").reset_index()
    sizes["n_authors"] = sizes["n_authors"].astype("Int64").mask(sizes["n_authors"] < MIN_SHOWN)
    clusters = f"{k} stable clusters"
    findings, refused = _compare_clusters(authors, cluster, k, settings, cs.top_drugs)
    said = statement(findings) if len(findings) else NOTHING_RELIABLE
    return SymptomSubgroups(
        clusters=clusters,
        reason=f"stability {choice.best:.2f}, tighter than with the links broken (p = {null_p:.2f})",
        k=k,
        summary=_describe(reported.loc[enough, list(symptoms)], cluster, cs.top_symptoms),
        sizes=sizes,
        null_p=null_p,
        findings=findings,
        refused=refused,
        statement=f"{clusters}; {said}",
        record=_record(data, [f"cluster {c}" for c in range(1, k + 1)], recorded),
        **base,
    )


def _no_clusters(data: SelfReportDataset, reason: str, null_p: float | None, **base) -> SymptomSubgroups:
    """The "no stable clusters" result: nothing to describe or test (spec S8)."""
    return SymptomSubgroups(
        clusters=NO_STABLE,
        reason=reason,
        k=None,
        summary=pd.DataFrame(columns=list(SUMMARY_COLUMNS)),
        sizes=pd.DataFrame({"cluster": [], "n_authors": pd.array([], dtype="Int64")}),
        null_p=null_p,
        findings=pd.DataFrame(columns=[*COLUMNS, "q_value", "label", "caveat"]),
        refused=pd.DataFrame(columns=list(REFUSED_COLUMNS)),
        statement=NO_STABLE,
        record=_record(data, [], base["settings"]),
        **base,
    )


def _compare_clusters(
    authors: pd.DataFrame, cluster: pd.Series, k: int, settings: Settings, top_drugs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each cluster against the other(s) by drug class, the `top_drugs` most common single drugs
    (`asclepius.profile.single_drugs`), onset during use and each trajectory, all corrected
    together and labelled; outcomes or sides with fewer than `settings.min_group_size` authors are
    refused (spec S4)."""
    minimum = settings.min_group_size
    outcomes: list[tuple[str, str, str, str, pd.Series]] = []  # comparison, level, id, name, yes/no or NA
    refused: list[dict[str, object]] = []
    has_class, has_drug = authors["classes"].map(bool), authors["drugs"].map(bool)
    groups = [(SUBGROUP_CLASS, CLASS, "class", cls, "classes", has_class) for cls in CLASSES]
    groups += [
        (SUBGROUP_DRUG, DRUG, "drug", drug, "drugs", has_drug)
        for drug in single_drugs(authors[has_drug], top_drugs)
    ]
    for comparison, level, prefix, name, column, among in groups:
        took = authors[column].map(lambda c, x=name: x in c)
        if int(took.sum()) < minimum:
            if took.any():
                refused.append(
                    {
                        "comparison": comparison,
                        "marker_id": f"{prefix}:{name}",
                        "marker_name": name,
                        "level": level,
                        "n_a": _shown(int(took.sum())),
                        "reason": f"fewer than {minimum} clustered authors took it (spec S4)",
                    }
                )
            continue
        outcomes.append((comparison, level, f"{prefix}:{name}", name, took.astype(float).where(among)))
    onset = authors["onset"].astype(object)
    outcomes.append(
        (
            SUBGROUP_ONSET,
            LEVEL,
            f"onset:{ONSET_DURING}",
            "onset during use",
            (onset == ONSET_DURING).astype(float).where(onset.isin([ONSET_DURING, ONSET_AFTER])),
        )
    )
    trajectory = authors["overall_trajectory"].astype(object)
    for value in TRAJECTORIES:
        yes = (trajectory == value).astype(float).where(trajectory.isin(TRAJECTORIES))
        outcomes.append((SUBGROUP_TRAJECTORY, LEVEL, f"trajectory:{value}", value, yes))

    sides = [(1, "cluster 2")] if k == 2 else [(c, "other clusters") for c in range(1, k + 1)]
    rows: list[dict[str, object]] = []
    for comparison, level, marker_id, name, values in outcomes:
        for c, other in sides:
            in_a = cluster == c
            a, b = values[in_a].dropna(), values[~in_a].dropna()
            head = {
                "comparison": comparison,
                "marker_id": marker_id,
                "marker_name": name,
                "level": level,
                "group_a": f"cluster {c}",
                "group_b": other,
            }
            if min(len(a), len(b)) < minimum:
                refused.append(
                    head
                    | {
                        "n_a": _shown(len(a)),
                        "n_b": _shown(len(b)),
                        "reason": f"fewer than {minimum} authors with a clear value on a side (spec S4)",
                    }
                )
                continue
            rows.append(head | {"sex": "all"} | compare_shares(int(a.sum()), len(a), int(b.sum()), len(b)))
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    findings = hide_small_cells(label_findings(table, settings, self_report=True))
    # with 3 or more clusters each row of an outcome shows its total and the other rows the rest;
    # and the trajectories add up to a side's size: hide each outcome, and all trajectories, together
    findings = hide_together(
        findings,
        findings["comparison"].where(findings["comparison"] == SUBGROUP_TRAJECTORY, findings["marker_id"]),
    )
    refused_table = pd.DataFrame(refused, columns=list(REFUSED_COLUMNS)).astype(
        {"n_a": "Int64", "n_b": "Int64"}
    )
    return findings, refused_table


def _record(data: SelfReportDataset, groups: list[str], settings: dict[str, object]) -> dict[str, object]:
    return run_record(data, groups, settings) | {"analysis": ANALYSIS, "note": NOTE}


def _describe(reported: pd.DataFrame, cluster: pd.Series, top: int) -> pd.DataFrame:
    """Each cluster's `top` symptoms with the largest share in it minus the share in the other
    authors, over-represented ones only. A symptom with fewer than 5 reporters or non-reporters in
    any cluster is not listed for any, since the shares and sizes would give that count back, with
    3 or more clusters through the other clusters' rows (spec S11)."""
    counts = reported.groupby(cluster.to_numpy()).agg(["sum", "count"])
    yes, size = counts.xs("sum", axis=1, level=1), counts.xs("count", axis=1, level=1)
    describable = [s for s in reported.columns if min(yes[s].min(), (size[s] - yes[s]).min()) >= MIN_SHOWN]
    reported = reported[describable]
    rows = []
    for c in sorted(cluster.unique()):
        inside = (cluster == c).to_numpy()
        n_in, n_out = int(inside.sum()), int((~inside).sum())
        listed = []
        for symptom in reported.columns:
            yes_in, yes_out = int(reported[symptom][inside].sum()), int(reported[symptom][~inside].sum())
            share_in, share_out = yes_in / n_in, yes_out / n_out
            if share_in > share_out:
                listed.append((share_in - share_out, symptom, share_in, share_out))
        for difference, symptom, share_in, share_out in sorted(listed, key=lambda r: (-r[0], r[1]))[:top]:
            rows.append(
                {
                    "cluster": int(c),
                    "n_authors": _shown(n_in),
                    "symptom": symptom,
                    "symptom_name": symptom.replace("_", " "),
                    "share_in": share_in,
                    "share_out": share_out,
                    "difference": difference,
                }
            )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)).astype({"n_authors": "Int64"})


def jaccard_distances(x: np.ndarray) -> np.ndarray:
    """Author-by-author Jaccard distance of yes/no rows: 1 - shared / either. Two rows with nothing
    reported are 0 apart."""
    x = x.astype(np.float32)
    shared = x @ x.T
    counts = x.sum(axis=1)
    either = counts[:, None] + counts[None, :] - shared
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(either > 0, 1 - shared / either, 0.0)
    return d.astype(np.float32)


def kmedoids(d: np.ndarray, k: int, restarts: int, rng: np.random.Generator) -> np.ndarray:
    """Labels 0..k-1 from the best (smallest total distance to the medoids) of `restarts` k-medoids
    runs on the distance matrix `d`, each started like k-means++ and improved by alternating
    assignment and medoid updates."""
    n = len(d)
    best_labels, best_cost = np.zeros(n, dtype=int), np.inf
    for _ in range(restarts):
        medoids = _starts(d, k, rng)
        for _ in range(100):
            labels = _assign(d, medoids)
            new = medoids.copy()
            for c in range(k):
                members = np.flatnonzero(labels == c)
                new[c] = members[d[np.ix_(members, members)].sum(axis=0).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        labels = _assign(d, medoids)
        cost = float(d[np.arange(n), medoids[labels]].sum())
        if cost < best_cost:
            best_labels, best_cost = labels, cost
    return best_labels


def _starts(d: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    medoids = [int(rng.integers(len(d)))]
    for _ in range(1, k):
        nearest = d[:, medoids].min(axis=1).astype(np.float64) ** 2
        total = nearest.sum()
        medoids.append(int(rng.choice(len(d), p=nearest / total)) if total > 0 else int(rng.integers(len(d))))
    return np.array(medoids)


def _assign(d: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """Each row's nearest medoid; a medoid always in its own cluster, even when tied."""
    labels = d[:, medoids].argmin(axis=1)
    labels[medoids] = np.arange(len(medoids))
    return labels


def spread(d: np.ndarray, labels: np.ndarray) -> float:
    """Total distance to the cluster medoids as a share of the total distance to the one best
    medoid of everyone: lower means tighter clusters."""
    within = 0.0
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        within += float(d[np.ix_(members, members)].sum(axis=0).min())
    return within / float(d.sum(axis=0).min())


def swap_randomise(x: np.ndarray, rounds: int, rng: np.random.Generator) -> np.ndarray:
    """`x` (authors by symptoms, yes/no) with the links between symptoms broken, keeping every row
    and column total: each round pairs every author with another, and each pair shuffles the
    symptoms only one of them reports between them, each keeping their number (curveball)."""
    x = x.astype(bool, copy=True)
    half = len(x) // 2
    for _ in range(rounds):
        order = rng.permutation(len(x))
        a, b = order[:half], order[half : 2 * half]
        xa, xb = x[a], x[b]
        both, differ = xa & xb, xa ^ xb
        keep = (xa & ~xb).sum(axis=1)  # how many of the differing symptoms row a has
        keys = np.where(differ, rng.random(xa.shape), 2.0)  # differing cells first, in random order
        rank = keys.argsort(axis=1).argsort(axis=1)
        to_a = differ & (rank < keep[:, None])
        x[a], x[b] = both | to_a, both | (differ & ~to_a)
    return x


def _null_p(
    x: np.ndarray, k: int, observed: float, cs: SymptomClusterSettings, rng: np.random.Generator
) -> float:
    """Share of swap-randomised datasets whose k-medoids `spread` is at most `observed` (counting
    the data itself, so never 0)."""
    tighter = 0
    for _ in range(cs.null_samples):
        d = jaccard_distances(swap_randomise(x, cs.swap_rounds, rng))
        tighter += spread(d, kmedoids(d, k, cs.restarts, rng)) <= observed
    return (tighter + 1) / (cs.null_samples + 1)


def _shown(count: int) -> int | None:
    return count if count >= MIN_SHOWN else None
