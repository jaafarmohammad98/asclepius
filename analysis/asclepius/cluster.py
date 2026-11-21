"""Subgroups of patients from their markers and enzyme-activity ratios (spec A3, S4, S8).

PSSD may be several conditions, so `cluster_patients` looks for subgroups among lab reports:

1. Every raw marker and every ratio (`asclepius.ratios`) is a feature. Each is turned into a
   normal score within sex (rank, then the normal quantile of rank / (n + 1)), because ranges
   differ by sex, and because ranks do not let one extreme value or a skewed marker dominate.
2. k-means (k-means++ starts, best of several) splits the patients into 2 to `k_max` clusters.
3. The number of clusters is the one whose clustering is most stable: patients are clustered again
   in random subsamples, and the adjusted Rand index says how well each subsample's clusters
   match the full clustering (1: the same, about 0: no better than chance).
4. Stability alone is not enough: k-means splits one elongated cloud of patients the same way
   every time. So the chosen clustering must also be tighter than k-means finds in data drawn
   from one normal distribution with the patients' own covariance (the idea of SigClust, Liu et al.
   2008). Failing either test gives "no stable clusters" (spec S8).

Thresholds were chosen without a clinician and are marked provisional until M3-11.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from asclepius.dataset import LabDataset
from asclepius.ratios import STATUS, compute_ratios

NO_STABLE = "no stable clusters"


@dataclass(frozen=True)
class ClusterSettings:
    """Every threshold `cluster_patients` uses; `record()` is what a run stores.

    Clustering is refused below `min_patients` (spec S4). A clustering is accepted when its
    stability (mean adjusted Rand index over `subsamples` subsamples of `subsample_share` of the
    patients) is at least `min_stability`, every cluster has at least `min_cluster_size` patients
    (the S4 group limit, so each cluster can be compared later), and its within-cluster spread is
    smaller than in all but `null_alpha` of `null_samples` one-group normal datasets. A feature
    missing for more than `max_missing_share` of the patients is left out; other missing values
    get the middle score (0).
    """

    min_patients: int = 30
    k_max: int = 6
    min_cluster_size: int = 10
    min_stability: float = 0.8
    subsamples: int = 50
    subsample_share: float = 0.8
    null_samples: int = 99
    null_alpha: float = 0.05
    restarts: int = 10
    max_missing_share: float = 0.2
    seed: int = 0
    status: str = field(default=STATUS)

    def __post_init__(self) -> None:
        if self.min_patients < 2 or self.min_cluster_size < 1:
            raise ValueError("min_patients must be at least 2 and min_cluster_size at least 1.")
        if self.k_max < 2:
            raise ValueError("k_max must be at least 2.")
        if not 0 < self.subsample_share < 1:
            raise ValueError("subsample_share must be between 0 and 1.")
        if not 0 <= self.min_stability <= 1 or not 0 < self.null_alpha <= 1:
            raise ValueError("min_stability must be between 0 and 1, and null_alpha above 0 and at most 1.")
        if min(self.subsamples, self.null_samples, self.restarts) < 1:
            raise ValueError("subsamples, null_samples and restarts must be at least 1.")
        if not 0 <= self.max_missing_share < 1:
            raise ValueError("max_missing_share must be at least 0 and below 1.")

    def record(self) -> dict[str, object]:
        return asdict(self)


class TooFewPatients(ValueError):
    """Clustering was refused: fewer patients than `ClusterSettings.min_patients` (spec S4)."""


@dataclass(frozen=True)
class ClusterResult:
    """What `cluster_patients` found.

    `statement`: "2 stable clusters" (or 3, ...) or "no stable clusters", with `reason` saying why.
    `k`: the number of clusters, None when there are none.
    `assignments`: `record` and `cluster` (1 is the largest), one row per patient. Working data
    for linking clusters to symptoms (M3-08b), never shown or exported (spec S11). Empty when
    there are no clusters.
    `sizes`: patients per `cluster` and `sex`. `profile`: per `cluster` and feature, the median
    normal score (0 is the middle of that sex, about +-1 one standard deviation), so a cluster can
    be described. `stability`: per `k`, its `stability`, `smallest` cluster and whether it was
    `eligible` (every cluster big enough). `null_p`: the one-group test's p-value for the chosen
    clustering. `features`: the marker and ratio ids used; `dropped`: features left out and how
    many patients lacked each; `filled`: missing values given the middle score; `no_sex`: reports
    left out for having no sex. `settings`: `ClusterSettings.record()`.
    """

    statement: str
    reason: str
    k: int | None
    assignments: pd.DataFrame
    sizes: pd.DataFrame
    profile: pd.DataFrame
    stability: pd.DataFrame
    null_p: float | None
    features: tuple[str, ...]
    dropped: pd.DataFrame
    filled: int
    no_sex: int
    settings: dict[str, object]


def cluster_patients(
    data: LabDataset, groups: Sequence[str] | None = None, settings: ClusterSettings | None = None
) -> ClusterResult:
    """Cluster the reports of `groups` (every report when None) by their markers and ratios.

    Reports with no sex are left out, since scores are made within sex. Refuses, with
    `TooFewPatients`, when fewer than `settings.min_patients` remain (spec S4). The same settings
    give the same result.
    """
    settings = settings or ClusterSettings()
    records = data.records
    if groups is not None:
        records = records[records["group"].isin(list(groups))]
    has_sex = records["sex"].notna()
    records = records[has_sex]
    n = len(records)
    if n < settings.min_patients:
        left_out = f" ({int((~has_sex).sum())} more left out for having no sex)" if not has_sex.all() else ""
        raise TooFewPatients(
            f"Refused: {n} patients{left_out}, and clustering needs at least {settings.min_patients} "
            f"(spec S4, {settings.status})."
        )
    scores, dropped, filled, names = _scores(data, records, settings)
    x = scores.to_numpy()
    rng = np.random.default_rng(settings.seed)

    rows, fits = [], {}
    for k in range(2, settings.k_max + 1):
        if k * settings.min_cluster_size > n:
            break
        labels, _ = _kmeans(x, k, settings.restarts, rng)
        fits[k] = labels
        smallest = int(np.bincount(labels, minlength=k).min())
        rows.append(
            {
                "k": k,
                "stability": _stability(x, labels, k, settings, rng),
                "smallest": smallest,
                "eligible": smallest >= settings.min_cluster_size,
            }
        )
    stability = pd.DataFrame(rows, columns=["k", "stability", "smallest", "eligible"])
    base = {
        "stability": stability,
        "features": tuple(scores.columns),
        "dropped": dropped,
        "filled": filled,
        "no_sex": int((~has_sex).sum()),
        "settings": settings.record(),
    }

    eligible = stability[stability["eligible"]]
    if eligible.empty:
        return _none(f"no split into clusters of at least {settings.min_cluster_size} patients", **base)
    best = eligible.loc[eligible["stability"].idxmax()]  # the first, so the smaller k, on a tie
    k = int(best["k"])
    if best["stability"] < settings.min_stability:
        return _none(
            f"the most stable split ({k} clusters) has stability {best['stability']:.2f}, "
            f"below {settings.min_stability}",
            **base,
        )
    sexes = records.set_index("record").loc[scores.index, "sex"].to_numpy()
    null_p = _null_p(x, sexes, k, _spread(x, fits[k]), settings, rng)
    if null_p > settings.null_alpha:
        return _none(
            f"{k} clusters are stable but no tighter than in one group with the same spread "
            f"(p = {null_p:.2f})",
            null_p=null_p,
            **base,
        )

    # number clusters by size, largest first; ties by the first patient in each
    labels = fits[k]
    order = sorted(range(k), key=lambda c: (-np.sum(labels == c), int(np.argmax(labels == c))))
    cluster = np.empty(k, dtype=int)
    cluster[order] = np.arange(1, k + 1)
    assignments = pd.DataFrame({"record": scores.index, "cluster": cluster[labels]})
    sizes = (
        assignments.assign(sex=sexes).groupby(["cluster", "sex"]).size().rename("n_patients").reset_index()
    )
    profile = (
        scores.groupby(assignments["cluster"].to_numpy())
        .median()
        .rename_axis("cluster")
        .reset_index()
        .melt(id_vars="cluster", var_name="marker_id", value_name="median_score")
    )
    profile.insert(2, "marker_name", profile["marker_id"].map(names))
    return ClusterResult(
        statement=f"{k} stable clusters",
        reason=f"stability {best['stability']:.2f}, tighter than one group (p = {null_p:.2f})",
        k=k,
        assignments=assignments,
        sizes=sizes,
        profile=profile,
        null_p=null_p,
        **base,
    )


def _none(reason: str, null_p: float | None = None, **base) -> ClusterResult:
    return ClusterResult(
        statement=NO_STABLE,
        reason=reason,
        k=None,
        assignments=pd.DataFrame(columns=["record", "cluster"]),
        sizes=pd.DataFrame(columns=["cluster", "sex", "n_patients"]),
        profile=pd.DataFrame(columns=["cluster", "marker_id", "marker_name", "median_score"]),
        null_p=null_p,
        **base,
    )


def _scores(
    data: LabDataset, records: pd.DataFrame, settings: ClusterSettings
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.Series]:
    """Patient-by-feature normal scores, made within sex, and the features dropped for missing too often.

    A limit ("<5") is read as its number, as the medians in `compare_groups` are.
    """
    values = pd.concat([data.markers, compute_ratios(data).data.markers], ignore_index=True)
    values = values[values["record"].isin(records["record"]) & values["value"].notna()]
    names = values.drop_duplicates("marker_id").set_index("marker_id")["marker_name"]
    table = values.pivot(index="record", columns="marker_id", values="value")
    table = table.reindex(index=records["record"], columns=sorted(table.columns))
    missing = table.isna().sum()
    too_often = missing > settings.max_missing_share * len(table)
    dropped = pd.DataFrame(
        {"marker_id": missing.index[too_often], "n_missing": missing[too_often].to_numpy()}
    )
    table = table.loc[:, ~too_often]
    if table.shape[1] < 2:
        raise ValueError("Clustering needs at least two markers that most patients have.")
    sex = records.set_index("record").loc[table.index, "sex"].to_numpy()
    filled = int(table.isna().sum().sum())
    scores = pd.DataFrame(_normal_scores(table.to_numpy(), sex), index=table.index, columns=table.columns)
    return scores.fillna(0.0), dropped, filled, names


def _normal_scores(values: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """Each column's normal scores within each sex: the normal quantile of rank / (n + 1), ties
    sharing their mean rank. Missing values stay missing."""
    ranks = pd.DataFrame(values).groupby(sex).rank(method="average").to_numpy()
    counts = pd.DataFrame(~np.isnan(values)).groupby(sex).transform("sum").to_numpy()
    out = np.full(values.shape, np.nan)
    known = ~np.isnan(ranks)
    out[known] = _QUANTILE(ranks[known] / (counts[known] + 1))
    return out


_QUANTILE = np.vectorize(NormalDist().inv_cdf, otypes=[float])


def _kmeans(x: np.ndarray, k: int, restarts: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """The best of `restarts` k-means runs from k-means++ starts: labels 0..k-1 and the
    within-cluster sum of squares."""
    best_labels, best_ss = np.zeros(len(x), dtype=int), np.inf
    for _ in range(restarts):
        centres = _plus_plus(x, k, rng)
        labels = np.full(len(x), -1)
        for _ in range(100):
            distance = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new = distance.argmin(axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                members = labels == c
                if members.any():
                    centres[c] = x[members].mean(axis=0)
                else:  # an empty cluster restarts at the point farthest from its centre
                    far = int(distance[np.arange(len(x)), labels].argmax())
                    centres[c], labels[far] = x[far], c
        ss = _within(x, labels, k)
        if ss < best_ss:
            best_labels, best_ss = labels.copy(), ss
    return best_labels, best_ss


def _plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        distance = np.min([((x - c) ** 2).sum(axis=1) for c in centres], axis=0)
        total = distance.sum()
        pick = rng.choice(len(x), p=distance / total) if total > 0 else rng.integers(len(x))
        centres.append(x[pick])
    return np.array(centres, dtype=float)


def _within(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    return float(
        sum(
            ((x[labels == c] - x[labels == c].mean(axis=0)) ** 2).sum()
            for c in range(k)
            if (labels == c).any()
        )
    )


def _spread(x: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squares as a share of the total, in the space the cluster centres span
    (for two clusters, the line through both centres): lower means tighter clusters.

    Measured there, not over every feature, because features the clusters do not differ on would
    swamp it: with 34 features and 60 patients, two clear subgroups barely change the total spread.
    """
    k = int(labels.max()) + 1
    centres = np.array([x[labels == c].mean(axis=0) for c in range(k)])
    basis = np.linalg.svd(centres - centres.mean(axis=0), full_matrices=False)[2][: k - 1]
    z = (x - x.mean(axis=0)) @ basis.T
    return _within(z, labels, k) / float((z**2).sum())


def _stability(x: np.ndarray, labels: np.ndarray, k: int, settings: ClusterSettings, rng) -> float:
    """Mean adjusted Rand index between the full clustering and each subsample's own clustering."""
    size = round(settings.subsample_share * len(x))
    agreement = []
    for _ in range(settings.subsamples):
        rows = rng.choice(len(x), size=size, replace=False)
        sub, _ = _kmeans(x[rows], k, settings.restarts, rng)
        agreement.append(adjusted_rand(labels[rows], sub))
    return float(np.mean(agreement))


def _null_p(x: np.ndarray, sex: np.ndarray, k: int, observed: float, settings: ClusterSettings, rng) -> float:
    """Share of one-group normal datasets, with the patients' covariance, whose k-means `_spread` is
    at most `observed` (counting the data itself, so never 0).

    Each sample is scored within sex, as the patients were. The sample covariance is used as it is. With about as many features as patients its largest
    directions are overstated, which makes one group look more clustered and the test cautious.
    """
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    scale = s / np.sqrt(len(x) - 1)
    tighter = 0
    for _ in range(settings.null_samples):
        # scored within sex like the patients, whose evenly spaced scores have lighter tails than
        # a normal sample and so split a little more cleanly
        null = _normal_scores((rng.standard_normal((len(x), len(s))) * scale) @ vt, sex)
        labels, _ = _kmeans(null, k, settings.restarts, rng)
        tighter += _spread(null, labels) <= observed
    return (tighter + 1) / (settings.null_samples + 1)


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labellings of the same patients (Hubert and Arabie 1985): 1 when
    they agree, about 0 when no better than chance."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(counts: np.ndarray) -> float:
        return float((counts * (counts - 1) / 2).sum())

    together, rows, columns = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * columns / pairs(np.array([len(a)]))
    most = (rows + columns) / 2
    return 1.0 if most == expected else (together - expected) / (most - expected)
