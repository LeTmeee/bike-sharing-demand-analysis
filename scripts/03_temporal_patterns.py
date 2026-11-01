"""RQ1：分析小时、星期、工作日、周末、年月与季节的需求规律。

读取第02步的 train_eda.csv，不清洗或修改输入。所有均值以实际观测
小时为分母；缺失分组保持为空，不补零。输出使用 03_ 前缀。
"""

# %% 参数与输入检查
import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/train_eda.csv")
    parser.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    return parser.parse_args()


def load_data(path, pd, np):
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["datetime", "count", "workingday", "holiday", "season"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"缺少字段：{sorted(missing)}；请重新运行第02步。")
    if data.empty:
        raise ValueError("输入数据为空。")
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dates = data["datetime"]
    if dates.isna().any() or dates.duplicated().any() or dates.ne(dates.dt.floor("h")).any():
        raise ValueError("时间戳存在缺失、不可解析、重复或非整点记录。")
    for field in required[1:]:
        values = pd.to_numeric(data[field], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any() or values.mod(1).ne(0).any():
            raise ValueError(f"{field} 存在缺失或非法数值。")
        allowed = {"workingday": [0, 1], "holiday": [0, 1], "season": [1, 2, 3, 4]}
        if field in allowed and not values.isin(allowed[field]).all():
            raise ValueError(f"{field} 存在非法类别。")
        data[field] = values.astype("int64")
    # 从 datetime 重新派生，避免手动修改后的时间字段与时间戳不一致。
    time = data["datetime"].dt
    data["date"] = time.strftime("%Y-%m-%d")
    data["year"] = time.year
    data["month"] = time.month
    data["year_month"] = time.strftime("%Y-%m")
    data["hour"] = time.hour
    data["weekday"] = time.dayofweek
    data["is_weekend"] = data["weekday"].ge(5).astype(int)
    return data.sort_values("datetime").reset_index(drop=True)


# %% 统一聚合：每张表都提供小时样本数、日期数、均值和中位数
def summarize(data, keys):
    return data.groupby(keys, observed=True, sort=True).agg(
        observed_hours=("count", "size"), observed_dates=("date", "nunique"),
        mean_count=("count", "mean"), median_count=("count", "median"),
        std_count=("count", "std"), q25_count=("count", lambda s: s.quantile(0.25)),
        q75_count=("count", lambda s: s.quantile(0.75)),
        observed_rental_sum=("count", "sum"),
    ).reset_index()


def build_tables(data, pd):
    groups = {
        "hour": ["hour"], "weekday": ["weekday"],
        "workingday": ["workingday"], "weekend": ["is_weekend"],
        "holiday": ["holiday"], "workingday_hour": ["workingday", "hour"],
        "weekend_hour": ["is_weekend", "hour"], "weekday_hour": ["weekday", "hour"],
        "year_month": ["year", "month", "year_month"],
        "season": ["season"], "year_season": ["year", "season"],
        "year_hour": ["year", "hour"],
    }
    tables = {name: summarize(data, keys) for name, keys in groups.items()}
    peaks = []
    for name in ["hour", "workingday_hour", "weekend_hour", "year_hour"]:
        table = tables[name]
        split = [key for key in groups[name] if key != "hour"]
        maximum = (table.groupby(split)["mean_count"].transform("max")
                   if split else table["mean_count"].max())
        # 并列最高小时全部保留，不任意取第一个。
        peak = table.loc[table["mean_count"].eq(maximum)].copy()
        peak.insert(0, "comparison", name)
        peaks.append(peak)
    tables["peak_hours"] = pd.concat(peaks, ignore_index=True)
    for metric, suffix in [("mean_count", "mean"), ("observed_hours", "samples")]:
        matrix = tables["weekday_hour"].pivot(index="weekday", columns="hour", values=metric)
        tables[f"weekday_hour_{suffix}_matrix"] = matrix.reindex(index=range(7), columns=range(24)).reset_index()
    return tables


# %% 图表：英文标签；空白/断线代表无观测，不解释为零需求
def save_figures(tables, directory, plt, sns):
    directory.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"03_{name}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    hourly = tables["hour"].set_index("hour").reindex(range(24))
    ax.plot(hourly.index, hourly["mean_count"], marker="o", label="Mean")
    ax.plot(hourly.index, hourly["median_count"], marker="s", label="Median")
    ax.set(xlabel="Hour", ylabel="Rentals per observed hour", title="Hourly rental demand", xticks=range(0, 24, 2))
    ax.legend()
    save(fig, "hourly_demand")

    for name, key, labels in [
        ("workingday_hour", "workingday", {0: "Non-working day", 1: "Working day"}),
        ("weekend_hour", "is_weekend", {0: "Mon-Fri", 1: "Sat-Sun"}),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        for value, part in tables[name].groupby(key):
            part = part.set_index("hour").reindex(range(24))
            ax.plot(part.index, part["mean_count"], marker="o", label=labels[value])
        ax.set(xlabel="Hour", ylabel="Mean rentals per observed hour", title=name.replace("_", " "), xticks=range(0, 24, 2))
        ax.legend()
        save(fig, name)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, suffix, title in zip(axes, ["mean", "samples"], ["Mean rentals per observed hour", "Observed hour sample size"]):
        matrix = tables[f"weekday_hour_{suffix}_matrix"].set_index("weekday")
        sns.heatmap(matrix, mask=matrix.isna(), cmap="YlGnBu", ax=ax,
                    yticklabels=WEEKDAYS, cbar_kws={"label": title})
        ax.set(title=title, xlabel="Hour", ylabel="Weekday")
    save(fig, "weekday_hour_heatmaps")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    weekday = tables["weekday"].set_index("weekday").reindex(range(7))
    axes[0].bar(WEEKDAYS, weekday["mean_count"])
    axes[0].set(title="Weekday demand", ylabel="Mean rentals per observed hour")
    axes[1].bar(WEEKDAYS, weekday["observed_hours"])
    axes[1].set(title="Weekday sample size", ylabel="Observed hours")
    save(fig, "weekday_summary")

    for name, key, full_range, title in [
        ("year_month", "month", range(1, 13), "Monthly demand by year (observed hours only)"),
        ("year_season", "season", range(1, 5), "Seasonal demand by year (original codes)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        for year, part in tables[name].groupby("year"):
            part = part.set_index(key).reindex(full_range)
            ax.plot(part.index, part["mean_count"], marker="o", label=str(year))
        ax.set(xlabel=key.capitalize(), ylabel="Mean rentals per observed hour", title=title, xticks=list(full_range))
        ax.legend(title="Year")
        save(fig, name)


# %% 主流程
def main():
    args = parse_args()
    if not args.input.is_file():
        print(f"缺少输入文件：{args.input}\n请先完成第01步核验并运行第02步。", file=sys.stderr)
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

    tables = build_tables(data, pd)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.tables_dir / f"03_{name}.csv", index=False, encoding="utf-8-sig")
    notes = [
        ("denominator", "均值、中位数及标准差基于已观测小时；std 使用样本标准差，单样本组为空。"),
        ("coverage", "训练集通常只有每月1–19日；observed_rental_sum不是完整月份或季节的总需求。"),
        ("missing", "缺失小时不补零；图表缺失分组留空，矩阵样本数为空代表该组合无观测。"),
        ("calendar", "workingday=0不能等同周末；周末由日期计算，holiday单独汇总。"),
        ("year", "年月与季节按年份分别展示；不能仅凭两年局部观测推断长期趋势。"),
        ("season", "保留原始1–4编码，不根据月份擅自重编码。"),
        ("peak", "峰值是各比较中小时均值最高的组；并列全部保留，需结合样本数解释。"),
        ("quality", "保留第02步所有复核标记对应记录，不自动删除极端值。"),
        ("interpretation", "描述性差异不证明通勤目的、因果关系或系统调度效果。"),
    ]
    pd.DataFrame(notes, columns=["topic", "note"]).to_csv(
        args.tables_dir / "03_interpretation_notes.csv", index=False, encoding="utf-8-sig")
    save_figures(tables, args.figures_dir, plt, sns)
    print(f"完成：分析 {len(data)} 个观测小时，生成 {len(tables) + 1} 张表及7张图。")
    print(f"表格：{args.tables_dir.resolve()}\n图片：{args.figures_dir.resolve()}")
    print("按实际观测小时比较需求；请结合样本数、年份与数据覆盖解释结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
