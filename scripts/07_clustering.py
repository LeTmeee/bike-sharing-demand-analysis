"""Daily demand-pattern KMeans analysis. See reports/clustering.md.

Only the 24 hourly count columns enter KMeans; metadata is post-hoc context.
Missing hours are never filled. Run from any working directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HOURS = [f"hour_{h:02d}" for h in range(24)]
CALENDAR = ["workingday", "holiday", "season"]
CONTINUOUS = ["temp", "atemp", "humidity", "windspeed"]


def build_daily(data):
    """Return count profiles and a date audit including entirely absent dates."""
    required = ["datetime", "count", *CALENDAR, "weather", *CONTINUOUS]
    if data.empty or not set(required).issubset(data):
        raise ValueError("Empty input or missing required columns: " + ", ".join(required))
    data = data[required].copy()  # Explicit allowlist excludes other target-derived columns.
    data["datetime"] = pd.to_datetime(data.datetime, errors="raise")
    t = data.datetime
    if t.dt.tz is not None or t.isna().any() or t.duplicated().any() or t.ne(t.dt.floor("h")).any():
        raise ValueError("Timestamps must be unique, naive, nonmissing whole hours")
    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError("Nonfinite values in " + col)
    if (data["count"].lt(0) | data["count"].mod(1).ne(0)).any():
        raise ValueError("Counts must be nonnegative integers")
    for field, values in dict(workingday=[0, 1], holiday=[0, 1],
                              season=[1, 2, 3, 4], weather=[1, 2, 3, 4]).items():
        if not data[field].isin(values).all():
            raise ValueError("Invalid category: " + field)
    if data[CONTINUOUS].lt(0).any().any() or data.humidity.gt(100).any():
        raise ValueError("Invalid weather variable range")
    data["date"] = t.dt.normalize()
    data["hour"] = t.dt.hour
    grouped = data.groupby("date", sort=True)
    if grouped[CALENDAR].nunique().gt(1).any().any():
        raise ValueError("Calendar metadata must be constant within each date")
    dates = pd.date_range(t.min().normalize(), t.max().normalize(), freq="D", name="date")
    profiles = data.pivot(index="date", columns="hour", values="count").reindex(index=dates, columns=range(24))
    profiles.columns = HOURS
    audit = pd.DataFrame(index=dates)
    audit["observed_hours"] = profiles.notna().sum(axis=1)
    audit["missing_hours"] = profiles.apply(lambda r: ";".join(str(h) for h in range(24) if pd.isna(r.iloc[h])), axis=1)
    audit["observed_total"] = profiles.sum(axis=1, min_count=1)
    audit["complete"] = audit.observed_hours.eq(24)
    audit["eligible"] = audit.complete & audit.observed_total.gt(0)
    audit["status"] = np.select([audit.observed_hours.eq(0), ~audit.complete,
                                 audit.observed_total.eq(0)],
                                ["no_target_observations", "incomplete", "zero_total"], default="included")
    for col in CALENDAR:
        audit[col] = grouped[col].first()
    audit["weather"] = grouped.weather.agg(lambda s: s.mode().iloc[0] if len(s.mode()) == 1 else np.nan)
    audit["weather_modal_share"] = grouped.weather.agg(lambda s: s.value_counts().max()/len(s))
    audit["weather_tied_mode"] = grouped.weather.agg(lambda s: len(s.mode()) > 1).astype("boolean")
    for col in CONTINUOUS:
        audit[col] = grouped[col].mean()
    audit["year"] = dates.year
    audit["month"] = dates.month
    audit["weekday"] = dates.dayofweek
    audit["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    return profiles, audit


def select_profiles(profiles, min_days=30, min_fraction=.5):
    """Conservative abstention guard; coverage denominator is observed dates."""
    selected = profiles.loc[profiles.notna().all(axis=1) & profiles.sum(axis=1).gt(0)].copy()
    observed = int(profiles.notna().any(axis=1).sum())
    if len(selected) < min_days or len(selected)/max(observed, 1) < min_fraction:
        raise ValueError(f"Insufficient complete positive-total days: {len(selected)}/{observed}; "
                         f"require >= {min_days} and fraction >= {min_fraction}. Audit saved. "
                         "Obtain additional observed complete profiles or design a separate "
                         "observed-hour summary analysis; do not fill missing demand with zero.")
    return selected


def representations(profiles):
    if profiles.isna().any().any() or profiles.sum(axis=1).le(0).any():
        raise ValueError("Representations require complete positive-total profiles")
    return {"raw": profiles.astype(float), "share": profiles.div(profiles.sum(axis=1), axis=0)}


def evaluate(x, max_k=6, seed=18, n_init=20):
    """Evaluate valid K values, selecting maximum silhouette (smaller K on ties)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < 3:
        raise ValueError("Need at least three finite profile rows")
    distinct = len(np.unique(x, axis=0))
    limit = min(max_k, len(x)-1, distinct)
    if limit < 2:
        raise ValueError("Need at least two distinct demand profiles")
    rows, models = [], {}
    # Small matrices: one numerical thread avoids overhead and improves repeatability.
    with threadpool_limits(limits=1):
        for k in range(1, limit+1):
            model = KMeans(n_clusters=k, init="k-means++", random_state=seed,
                           n_init=n_init, algorithm="lloyd", max_iter=300, tol=1e-4).fit(x)
            actual = len(np.unique(model.labels_))
            score = silhouette_score(x, model.labels_) if 2 <= actual < len(x) else np.nan
            rows.append(dict(k=k, actual_clusters=actual, inertia=model.inertia_,
                             silhouette=score, min_cluster_size=int(np.bincount(model.labels_).min()),
                             iterations=model.n_iter_))
            models[k] = model
    metrics = pd.DataFrame(rows)
    valid = metrics.loc[metrics.k.ge(2) & metrics.actual_clusters.eq(metrics.k)].dropna(subset=["silhouette"])
    if valid.empty:
        raise ValueError("No valid candidate clustering")
    best = int(valid.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    metrics["inertia_reduction_fraction"] = 1 - metrics.inertia/metrics.inertia.shift(1)
    metrics["selected"] = metrics.k.eq(best)
    return metrics, models, best


def metadata_distribution(audit, labels):
    """Within-cluster proportions; ambiguous weather modes remain a category."""
    rows = []
    for field in [*CALENDAR, "weather", "year", "month", "weekday", "is_weekend"]:
        values = audit[field].astype("Int64").astype("string").fillna("ambiguous")
        for cluster in sorted(labels.unique()):
            group = values.loc[labels.eq(cluster)]
            for category, count in group.value_counts().sort_index().items():
                rows.append(dict(cluster=int(cluster), field=field, category=category,
                                 days=int(count), within_cluster_share=count/len(group)))
    return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clusters(profiles, shares, labels, distributions, name, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = labels.value_counts().sort_index()
    for cluster, size in counts.items():
        mask = labels.eq(cluster)
        for ax, matrix in zip(axes, [profiles, shares]):
            ax.plot(range(24), matrix.loc[mask].mean(), marker=".", label=f"C{cluster} (n={size})")
    for ax, ylabel in zip(axes, ["Mean hourly rentals", "Mean within-day share"]):
        ax.set(xlabel="Hour", ylabel=ylabel, xticks=range(0, 24, 3), title=f"{name}: complete days")
        ax.grid(alpha=.2)
        ax.legend()
    save_figure(fig, directory / f"07_{name}_profiles.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#3577A8")
    for i, size in enumerate(counts):
        ax.text(i, size, str(size), ha="center", va="bottom")
    ax.set(xlabel="Cluster", ylabel="Complete dates", title=f"{name}: cluster sizes", ylim=(0, counts.max()*1.15))
    save_figure(fig, directory / f"07_{name}_sizes.png")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, field in zip(axes.flat, ["workingday", "season", "weather", "year", "month", "weekday"]):
        subset = distributions.loc[distributions.field.eq(field)]
        table = subset.pivot(index="cluster", columns="category", values="within_cluster_share").fillna(0)
        table = table.reindex(columns=sorted(table.columns, key=lambda c: (c == "ambiguous", int(c) if c != "ambiguous" else 0)))
        table.plot.bar(stacked=True, ax=ax, colormap="tab20", rot=0)
        ax.set(title=f"{field} (source codes)", ylabel="Within-cluster share", ylim=(0, 1.02))
        ax.legend(title="Code", fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18))
    save_figure(fig, directory / f"07_{name}_metadata.png")


def run(args):
    data = pd.read_csv(args.input)
    profiles, audit = build_daily(data)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.tables_dir / "07_day_audit.csv", encoding="utf-8-sig")
    profiles.to_csv(args.tables_dir / "07_profiles_all_dates.csv", encoding="utf-8-sig")
    # Audit is saved before the sufficiency guard, including on an aborted run.
    selected = select_profiles(profiles)
    matrices = representations(selected)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    context = audit.loc[selected.index].copy()
    context["daily_total"] = selected.sum(axis=1)
    assignments, summaries, metrics_list, distribution_list, profile_list, centroid_list = [], [], [], [], [], []
    final_labels, choices = {}, {}
    for name, matrix in matrices.items():
        metrics, models, k = evaluate(matrix, args.max_k, args.seed, args.n_init)
        # Fit a final model explicitly using the chosen settings and the same ordered data.
        with threadpool_limits(limits=1):
            final = KMeans(**models[k].get_params()).fit(matrix.to_numpy())
        # Numeric IDs ordered by mean daily total for readable, reproducible outputs.
        totals = context.daily_total.groupby(final.labels_).mean().sort_values(kind="stable")
        mapping = {int(old): new for new, old in enumerate(totals.index)}
        labels = pd.Series([mapping[int(v)] for v in final.labels_], index=matrix.index, name="cluster")
        final_labels[name] = labels
        choices[name] = dict(k=k, silhouette=float(metrics.loc[metrics.k.eq(k), "silhouette"].iloc[0]),
                             inertia=float(final.inertia_), label_mapping=mapping)
        metrics_list.append(metrics.assign(representation=name))
        assignments.append(context.assign(cluster=labels, representation=name).reset_index())
        distributions = metadata_distribution(context, labels)
        distribution_list.append(distributions.assign(representation=name))
        for cluster in sorted(labels.unique()):
            mask = labels.eq(cluster)
            group = context.loc[mask]
            count_mean = selected.loc[mask].mean()
            share_mean = matrices["share"].loc[mask].mean()
            summaries.append(dict(representation=name, cluster=int(cluster), days=len(group),
                                  fraction=len(group)/len(context), mean_daily_total=group.daily_total.mean(),
                                  median_daily_total=group.daily_total.median(),
                                  workingday_share=group.workingday.mean(),
                                  peak_hours=";".join(str(h) for h in np.flatnonzero(count_mean.eq(count_mean.max()))),
                                  morning_share=share_mean.iloc[7:10].sum(),
                                  evening_share=share_mean.iloc[16:20].sum(),
                                  midday_share=share_mean.iloc[10:16].sum(),
                                  **{f"mean_{col}": group[col].mean() for col in CONTINUOUS}))
            for h in range(24):
                profile_list.append(dict(representation=name, cluster=int(cluster), hour=h,
                                         mean_count=count_mean.iloc[h], mean_share=share_mean.iloc[h]))
        for old, center in enumerate(final.cluster_centers_):
            centroid_list.append(dict(representation=name, cluster=mapping[old], **dict(zip(HOURS, center))))
        plot_clusters(selected, matrices["share"], labels, distributions, name, args.figures_dir)
    metrics = pd.concat(metrics_list, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, name in enumerate(matrices):
        values = metrics.loc[metrics.representation.eq(name)]
        for ax, field in zip(axes[row], ["inertia", "silhouette"]):
            ax.plot(values.k, values[field], marker="o")
            ax.axvline(choices[name]["k"], color="gray", linestyle="--", label="Selected K")
            ax.set(xlabel="K", ylabel=field, title=f"{name}: {field}", xticks=values.k)
            ax.legend()
    save_figure(fig, args.figures_dir / "07_k_selection.png")
    observed = audit.loc[audit.observed_hours.gt(0)]
    retention = []
    for field in [*CALENDAR, "weather", "year", "month"]:
        for category, group in observed.groupby(field, dropna=False):
            retention.append(dict(field=field, category=category, observed_days=len(group),
                                  included_days=int(group.eligible.sum()), retained_fraction=group.eligible.mean()))
    comparison = pd.crosstab(final_labels["raw"], final_labels["share"],
                             rownames=["raw_cluster"], colnames=["share_cluster"])
    comparison.to_csv(args.tables_dir / "07_cluster_crosstab.csv", encoding="utf-8-sig")
    tables = dict(k_metrics=metrics, assignments=pd.concat(assignments, ignore_index=True),
                  cluster_summary=pd.DataFrame(summaries), metadata_distribution=pd.concat(distribution_list, ignore_index=True),
                  mean_profiles=pd.DataFrame(profile_list), centroids=pd.DataFrame(centroid_list),
                  retention=pd.DataFrame(retention))
    for name, table in tables.items():
        table.to_csv(args.tables_dir / f"07_{name}.csv", index=False, encoding="utf-8-sig")
    for name, matrix in matrices.items():
        matrix.to_csv(args.tables_dir / f"07_{name}_features.csv", encoding="utf-8-sig")
    metadata = dict(input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    seed=args.seed, n_init=args.n_init, max_k=args.max_k, primary="share", scaling="none; equal hourly weights",
                    feature_columns=HOURS, selection_rule="maximum full-sample silhouette; smaller K on exact tie",
                    min_complete_days=30, min_complete_fraction=.5, numerical_threads=1,
                    calendar_days=len(audit), observed_days=len(observed), complete_days=int(audit.complete.sum()),
                    incomplete_days=int((observed.observed_hours.lt(24)).sum()),
                    absent_dates=int(audit.observed_hours.eq(0).sum()), included_days=len(selected),
                    zero_total_complete_days=int((audit.complete & audit.observed_total.eq(0)).sum()),
                    selected=choices, representation_adjusted_rand=float(adjusted_rand_score(final_labels["raw"], final_labels["share"])),
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                                  sklearn=sklearn.__version__, matplotlib=matplotlib.__version__))
    (args.tables_dir / "07_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(tables["cluster_summary"].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train_eda.csv")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-k", type=int, default=6)
    args = parser.parse_args()
    if args.max_k < 2 or args.n_init < 1:
        parser.error("--max-k must be >=2 and --n-init must be positive")
    # Refuse an output/input name collision even for custom input paths.
    if args.input.name.startswith("07_") and args.input.resolve().parent in [args.tables_dir.resolve(), args.figures_dir.resolve()]:
        parser.error("Input must be separate from generated 07_* outputs")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        print(f"Clustering stopped: {exc}. Existing outputs may be from an older run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
