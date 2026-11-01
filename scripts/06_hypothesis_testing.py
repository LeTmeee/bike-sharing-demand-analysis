"""Daily hypothesis tests; run from any directory. See reports/hypothesis_testing.md."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def aggregate(data):
    required = {"datetime", "count", "workingday", "weather", "temp"}
    if not required.issubset(data) or data.empty:
        raise ValueError("Empty input or missing required columns")
    data = data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    t = data.datetime
    if t.isna().any() or t.duplicated().any() or t.ne(t.dt.floor("h")).any():
        raise ValueError("Timestamps must be unique, nonmissing whole hours")
    for col in required - {"datetime"}:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError("Nonfinite values in " + col)
    if (data["count"].lt(0) | data["count"].mod(1).ne(0)).any():
        raise ValueError("Counts must be nonnegative integers")
    if not data.workingday.isin([0, 1]).all() or not data.weather.isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid category codes")
    data["date"] = t.dt.normalize()
    g = data.groupby("date", sort=True)
    if g.workingday.nunique().gt(1).any():
        raise ValueError("Working-day flag varies within a date")
    daily = g.agg(observed_hours=("count", "size"), demand=("count", "mean"),
                  observed_total=("count", "sum"), temp=("temp", "mean"),
                  workingday=("workingday", "first"))
    # Never break a weather tie by arbitrarily selecting the smallest code.
    daily["weather"] = g.weather.agg(lambda x: x.mode().iloc[0] if len(x.mode()) == 1 else np.nan)
    daily["weather_modal_share"] = g.weather.agg(lambda x: x.value_counts().max() / len(x))
    return daily.reset_index()


def delta(x, y):
    return 2 * stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic / (len(x) * len(y)) - 1


def interval(rng, x, y, fn, reps, paired=False):
    values = []
    for _ in range(reps):
        ix = rng.integers(len(x), size=len(x))
        iy = ix if paired else rng.integers(len(y), size=len(y))
        values.append(fn(x[ix], y[iy]))
    valid = np.asarray(values)[np.isfinite(values)]
    if len(valid) < 0.95 * reps:
        return np.nan, np.nan
    return tuple(np.quantile(valid, [0.025, 0.975]))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def analyze(d, label, rng, reps):
    rows, posthoc = [], []
    x = d.loc[d.workingday.eq(1), "demand"].to_numpy()
    y = d.loc[d.workingday.eq(0), "demand"].to_numpy()
    if min(len(x), len(y)) < 2:
        raise ValueError("At least two dates per working-day group required")
    u = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    lo, hi = interval(rng, x, y, delta, reps)
    rows.append(dict(analysis=label, test="workingday_Mann_Whitney", n=len(x)+len(y),
                     statistic=u.statistic, p_value=u.pvalue, effect="Cliffs_delta_work_minus_nonwork",
                     estimate=delta(x, y), ci_low=lo, ci_high=hi))
    groups = {int(k): p.demand.to_numpy() for k, p in d.dropna(subset=["weather"]).groupby("weather")}
    eligible = {k: v for k, v in groups.items() if len(v) >= 5}
    if len(eligible) >= 2:
        h = stats.kruskal(*eligible.values())
        n = sum(map(len, eligible.values()))
        # Rank epsilon squared, truncated at zero; no CI claimed.
        eps = max(0, (h.statistic - len(eligible) + 1) / (n - len(eligible)))
        rows.append(dict(analysis=label, test="weather_Kruskal_Wallis", n=n,
                         statistic=h.statistic, p_value=h.pvalue, effect="rank_epsilon_squared",
                         estimate=eps, ci_low=np.nan, ci_high=np.nan))
        if h.pvalue < .05:
            for a, b in combinations(eligible, 2):
                xa, xb = eligible[a], eligible[b]
                result = stats.mannwhitneyu(xa, xb, alternative="two-sided", method="asymptotic")
                lo, hi = interval(rng, xa, xb, delta, reps)
                posthoc.append(dict(analysis=label, weather_a=a, weather_b=b, n_a=len(xa), n_b=len(xb),
                                    p_value=result.pvalue, cliffs_delta=delta(xa, xb), ci_low=lo, ci_high=hi))
            adjusted = holm([r["p_value"] for r in posthoc])
            for row, p in zip(posthoc, adjusted):
                row["p_holm"] = p
    x, y = d.temp.to_numpy(), d.demand.to_numpy()
    r = stats.spearmanr(x, y)
    lo, hi = interval(rng, x, y, lambda a, b: stats.spearmanr(a, b).statistic, reps, paired=True)
    rows.append(dict(analysis=label, test="temperature_Spearman", n=len(d), statistic=r.statistic,
                     p_value=r.pvalue, effect="Spearman_rho", estimate=r.statistic, ci_low=lo, ci_high=hi))
    return rows, posthoc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/processed/train_eda.csv")
    p.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/tables")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=18)
    args = p.parse_args()
    if args.bootstrap < 200:
        p.error("--bootstrap must be at least 200")
    daily = aggregate(pd.read_csv(args.input, encoding="utf-8-sig"))
    primary = daily.loc[daily.observed_hours.eq(24)].copy()
    rng = np.random.default_rng(args.seed)
    rows, post = [], []
    summaries = []
    for label, d in [("complete_days_primary", primary), ("all_days_sensitivity", daily)]:
        r, ph = analyze(d, label, rng, args.bootstrap)
        rows.extend(r)
        post.extend(ph)
        for field in ["workingday", "weather"]:
            for code in ([0, 1] if field == "workingday" else [1, 2, 3, 4]):
                v = d.loc[d[field].eq(code), "demand"]
                summaries.append(dict(analysis=label, group=field, code=code, n_dates=len(v),
                                      mean=v.mean(), median=v.median(), sd=v.std(),
                                      eligible_for_weather_test=(len(v) >= 5) if field == "weather" else True))
    tests = pd.DataFrame(rows)
    tests["p_holm_three_tests"] = np.nan
    for label in tests.analysis.unique():
        mask = tests.analysis.eq(label)
        tests.loc[mask, "p_holm_three_tests"] = holm(tests.loc[mask, "p_value"])
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in {"daily": daily, "tests": tests, "group_summary": pd.DataFrame(summaries),
                        "weather_posthoc": pd.DataFrame(post, columns=["analysis", "weather_a", "weather_b", "n_a", "n_b", "p_value", "cliffs_delta", "ci_low", "ci_high", "p_holm"])}.items():
        table.to_csv(args.tables_dir / f"06_{name}.csv", index=False, encoding="utf-8-sig")
    metadata = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    seed=args.seed, bootstrap=args.bootstrap, python=platform.python_version(),
                    pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__,
                    dates=len(daily), complete_dates=len(primary), tied_weather_dates=int(daily.weather.isna().sum()))
    (args.tables_dir / "06_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, field, codes in [(axes[0], "workingday", [0, 1]), (axes[1], "weather", sorted(primary.weather.dropna().unique()))]:
        values = [primary.loc[primary[field].eq(c), "demand"] for c in codes]
        ax.boxplot(values)
        ax.set_xticks(range(1, len(codes) + 1))
        ax.set_xticklabels([f"{int(c)} (n={len(v)})" for c, v in zip(codes, values)])
        ax.set(xlabel=field, ylabel="Mean hourly rentals within date", title="Complete dates")
    axes[2].scatter(primary.temp, primary.demand, s=12, alpha=.5)
    axes[2].set(xlabel="Daily mean temp (source units)", ylabel="Mean hourly rentals within date", title="Complete dates: temperature")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "06_daily_hypothesis_tests.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
