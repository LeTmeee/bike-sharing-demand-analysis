"""RQ2：天气类别、连续天气变量与每小时租赁需求的描述性关联。

读取第02步的 train_eda.csv，不修改输入或自动排除复核记录。
保留天气原始编码；分层比较和零值敏感性分析不能解释为因果估计。
"""

# %% 路径与参数
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WEATHER = ["temp", "atemp", "humidity", "windspeed"]
CORR_FIELDS = WEATHER + ["count"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train_eda.csv")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="稀疏组提示阈值，默认30；不是显著性或可靠性标准")
    return parser.parse_args()


# %% 输入校验：独立运行，不依赖前一步内存或旧审计结果
def load_data(path, pd, np):
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["datetime", "weather", "season", "workingday", *CORR_FIELDS]
    missing = set(required) - set(data.columns)
    if missing or data.empty:
        raise ValueError(f"输入为空或缺少字段：{sorted(missing)}；请重新运行第02步。")
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dates = data["datetime"]
    if dates.isna().any() or dates.duplicated().any() or dates.ne(dates.dt.floor("h")).any():
        raise ValueError("时间戳存在缺失、重复、不可解析或非整点记录。")
    allowed = {"weather": [1, 2, 3, 4], "season": [1, 2, 3, 4], "workingday": [0, 1]}
    for field in required[1:]:
        values = pd.to_numeric(data[field], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"{field} 存在缺失、非有限值或负数。")
        if field in allowed and not values.isin(allowed[field]).all():
            raise ValueError(f"{field} 存在非法类别编码。")
        if field == "humidity" and values.gt(100).any():
            raise ValueError("humidity 超出0–100范围。")
        if field == "count" and values.mod(1).ne(0).any():
            raise ValueError("count 必须为非负整数。")
        data[field] = values.astype(int) if field in allowed or field == "count" else values
    data["hour"] = dates.dt.hour
    data["year"] = dates.dt.year
    data["date"] = dates.dt.strftime("%Y-%m-%d")
    return data.sort_values("datetime").reset_index(drop=True)


def summarize(data, keys, minimum):
    result = data.groupby(keys, observed=True, sort=True).agg(
        observed_hours=("count", "size"), observed_dates=("date", "nunique"),
        mean_count=("count", "mean"), median_count=("count", "median"),
        std_count=("count", "std"), q25_count=("count", lambda s: s.quantile(0.25)),
        q75_count=("count", lambda s: s.quantile(0.75)),
    ).reset_index()
    result["sparse_group"] = result["observed_hours"].lt(minimum).astype(int)
    return result


def correlation_pair(data, field):
    """秩变换后计算 Pearson 得到 Spearman，无需额外安装 SciPy。"""
    x, y = data[field], data["count"]
    if len(data) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    return x.corr(y), x.rank(method="average").corr(y.rank(method="average"))


# %% 表格：类别、分层、分箱、相关性与零值敏感性
def build_tables(data, pd, np, minimum):
    tables = {}
    for name, keys in {
        "weather_summary": ["weather"], "weather_hour": ["weather", "hour"],
        "weather_workingday": ["workingday", "weather"],
        "weather_season": ["season", "weather"],
        "weather_year": ["year", "weather"],
        "weather_hour_workingday": ["workingday", "weather", "hour"],
    }.items():
        tables[name] = summarize(data, keys, minimum)

    bins, edges_table = [], []
    for field in WEATHER:
        lower, upper = data[field].min(), data[field].max()
        # 常量变量只生成一个区间，避免重复边界。
        edges = np.linspace(lower, upper, 9) if lower < upper else np.array([lower - 0.5, upper + 0.5])
        work = data.copy()
        work["bin_id"] = pd.cut(work[field], bins=edges, labels=False, include_lowest=True)
        summary = summarize(work, ["bin_id"], minimum).set_index("bin_id").reindex(range(len(edges) - 1))
        summary["observed_hours"] = summary["observed_hours"].fillna(0).astype(int)
        summary["observed_dates"] = summary["observed_dates"].fillna(0).astype(int)
        summary["sparse_group"] = summary["observed_hours"].lt(minimum).astype(int)
        summary["x_mean"] = work.groupby("bin_id")[field].mean()
        summary["field"] = field
        summary["left_edge"], summary["right_edge"] = edges[:-1], edges[1:]
        bins.append(summary.reset_index())
        for i in range(len(edges) - 1):
            edges_table.append(dict(field=field, bin_id=i, left_edge=edges[i], right_edge=edges[i + 1]))
    tables["binned_demand"] = pd.concat(bins, ignore_index=True)
    tables["bin_edges"] = pd.DataFrame(edges_table)

    for method in ["pearson", "spearman"]:
        values = data[CORR_FIELDS] if method == "pearson" else data[CORR_FIELDS].rank(method="average")
        tables[f"correlation_{method}"] = values.corr().rename_axis("field").reset_index()

    associations = []
    # 分层逐项观察关联变化，不表示已经控制所有混杂因素。
    subsets = [("all", "all", data)]
    for key in ["workingday", "season", "hour", "year"]:
        subsets.extend((key, str(value), part) for value, part in data.groupby(key))
    for dimension, group, part in subsets:
        for field in WEATHER:
            pearson, spearman = correlation_pair(part, field)
            associations.append(dict(stratum=dimension, group=group, field=field,
                                     observed_hours=len(part), pearson=pearson, spearman=spearman,
                                     sparse_group=int(len(part) < minimum)))
    tables["stratified_correlations"] = pd.DataFrame(associations)
    sensitivity = []
    for field in ["humidity", "windspeed"]:
        for label, part in [("all", data), ("positive_only", data.loc[data[field].gt(0)])]:
            pearson, spearman = correlation_pair(part, field)
            sensitivity.append(dict(field=field, subset=label, observed_hours=len(part),
                                    excluded_hours=len(data) - len(part), pearson=pearson,
                                    spearman=spearman, sparse_group=int(len(part) < minimum)))
    tables["zero_sensitivity"] = pd.DataFrame(sensitivity)
    return tables


# %% 六张图：保留全样本；缺失分组不填需求零值
def save_figures(data, tables, directory, plt, sns):
    directory.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"04_{name}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=data, x="weather", y="count", order=[1, 2, 3, 4], ax=axes[0])
    axes[0].set(title="Demand by weather (original codes)", ylabel="Hourly rentals")
    samples = tables["weather_summary"].set_index("weather").reindex([1, 2, 3, 4])
    axes[1].bar([1, 2, 3, 4], samples["observed_hours"].fillna(0))
    axes[1].set(title="Weather sample size", xlabel="Weather code", ylabel="Observed hours", xticks=[1, 2, 3, 4])
    save(fig, "weather_categories")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, field in zip(axes.flat, WEATHER):
        ax.scatter(data[field], data["count"], s=8, alpha=0.12, rasterized=True)
        ax.set(xlabel=field, ylabel="Hourly rentals", title=f"{field} vs demand (all observations)")
    save(fig, "weather_scatterplots")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, field in zip(axes.flat, WEATHER):
        part = tables["binned_demand"].query("field == @field")
        # 固定区间中点为横坐标，空区间均值保持 NaN，曲线自然断开。
        x = (part["left_edge"] + part["right_edge"]) / 2
        ax.plot(x, part["mean_count"], marker="o", label="Mean")
        ax.plot(x, part["median_count"], marker="s", label="Median")
        for xpos, (_, row) in zip(x, part.iterrows()):
            if row["observed_hours"]:
                ax.annotate(f"n={int(row['observed_hours'])}", (xpos, row["mean_count"]), fontsize=7)
        ax.set(xlabel=f"{field}: equal-width bin midpoint", ylabel="Rentals per observed hour")
        ax.legend()
    save(fig, "binned_demand")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, metric, title in zip(axes, ["mean_count", "observed_hours"], ["Mean hourly rentals", "Observed hours"]):
        matrix = tables["weather_hour"].pivot(index="weather", columns="hour", values=metric).reindex(index=[1, 2, 3, 4], columns=range(24))
        sns.heatmap(matrix, mask=matrix.isna(), cmap="YlGnBu", ax=ax)
        ax.set(title=title, xlabel="Hour", ylabel="Weather code")
    save(fig, "weather_hour_heatmaps")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ["pearson", "spearman"]):
        matrix = tables[f"correlation_{method}"].set_index("field")
        sns.heatmap(matrix, mask=matrix.isna(), annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=ax)
        ax.set_title(f"{method.capitalize()} association")
    save(fig, "correlations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, name in [(axes[0], "workingday", "weather_workingday"), (axes[1], "season", "weather_season")]:
        for value, part in tables[name].groupby(key):
            part = part.set_index("weather").reindex([1, 2, 3, 4])
            ax.plot(part.index, part["mean_count"], marker="o", label=f"{key}={value}")
        ax.set(xlabel="Weather code", ylabel="Mean rentals per observed hour", title=f"Weather by {key}", xticks=[1, 2, 3, 4])
        ax.legend()
    save(fig, "stratified_weather")


def main():
    args = parse_args()
    if args.min_samples < 1:
        print("--min-samples 必须是正整数。", file=sys.stderr)
        return 1
    if not args.input.is_file():
        print(f"缺少输入：{args.input}\n请先完成第01、02步。", file=sys.stderr)
        return 1
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        print(f"缺少依赖：{exc}\n请运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    try:
        data = load_data(args.input, pd, np)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"输入检查失败：{exc}\n本次未更新输出；旧文件不代表本次结果。", file=sys.stderr)
        return 1
    tables = build_tables(data, pd, np, args.min_samples)
    notes = [
        ("scope", "天气与需求为描述性关联；时间、季节和年份可能相互混杂，分层不等于因果控制。"),
        ("categories", "weather和season保留原始编码，不把类别数值加入连续变量相关矩阵。"),
        ("sparse", f"样本数小于{args.min_samples}标记为稀疏组；阈值仅作提示，不是统计显著性标准。"),
        ("bins", "连续变量按全样本范围分8个等宽区间；常量变量单独一个区间。区间右闭，首区间包含最小值。"),
        ("empty_bins", "空区间观测数为0，需求统计为空；不会将无观测解释为零需求。"),
        ("correlation", "Pearson衡量线性关联；Spearman通过平均秩计算。样本不足或任一变量为常量时相关系数为空。"),
        ("sensitivity", "主分析保留全部数据；仅在敏感性表分别排除湿度零值或风速零值，不修改分析数据。"),
        ("sampling", "训练集仅覆盖各月部分日期；均值基于观测小时，不推断完整月份总量或长期规律。"),
        ("units", "图中保留原始字段名和数值；单位需核对数据字典，temp与atemp的相关性也需关注。"),
    ]
    tables["interpretation_notes"] = pd.DataFrame(notes, columns=["topic", "note"])
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.tables_dir / f"04_{name}.csv", index=False, encoding="utf-8-sig")
    save_figures(data, tables, args.figures_dir, plt, sns)
    print(f"完成：{len(data)}个观测小时，{len(tables)}张表、6张图。")
    print(f"表格：{args.tables_dir.resolve()}\n图片：{args.figures_dir.resolve()}")
    print("请结合样本数和分层结果解释天气关联，不作因果推断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
