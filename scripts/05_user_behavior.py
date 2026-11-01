"""RQ3：注册与临时用户的租赁数量、构成及时间模式。

分组占比=该组某类租赁量总和/该组总租赁量，不平均逐小时占比。
租赁次数不等于独立用户人数；不根据注册类别推断出行目的。
"""

# %% 参数与输入
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
USERS = ["casual", "registered"]
COLORS = {"casual": "#DB8A35", "registered": "#3577A8"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train_eda.csv")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    return parser.parse_args()


def load_data(path, pd, np):
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["datetime", "count", *USERS, "workingday", "holiday", "season"]
    missing = set(required) - set(data.columns)
    if missing or data.empty:
        raise ValueError(f"输入为空或缺少字段：{sorted(missing)}。")
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dates = data["datetime"]
    if dates.isna().any() or dates.duplicated().any() or dates.ne(dates.dt.floor("h")).any():
        raise ValueError("时间戳存在缺失、重复、不可解析或非整点记录。")
    allowed = {"workingday": [0, 1], "holiday": [0, 1], "season": [1, 2, 3, 4]}
    for field in required[1:]:
        values = pd.to_numeric(data[field], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any() or values.mod(1).ne(0).any():
            raise ValueError(f"{field} 存在缺失、非有限值、负数或非整数。")
        if field in allowed and not values.isin(allowed[field]).all():
            raise ValueError(f"{field} 存在非法类别。")
        data[field] = values.astype("int64")
    if data["count"].ne(data["casual"] + data["registered"]).any():
        raise ValueError("存在 count != casual + registered 的记录。")
    data["date"] = dates.dt.strftime("%Y-%m-%d")
    data["hour"] = dates.dt.hour
    data["weekday"] = dates.dt.day_of_week
    data["is_weekend"] = data["weekday"].ge(5).astype(int)
    data["year"] = dates.dt.year
    data["month"] = dates.dt.month
    data["year_month"] = dates.dt.strftime("%Y-%m")
    data["scope"] = "all"
    return data.sort_values("datetime").reset_index(drop=True)


# %% 统一长表：每一组分别报告两类用户，分母保持一致
def summarize(data, keys, pd):
    parts = []
    for user in USERS:
        part = data.groupby(keys, observed=True, sort=True).agg(
            observed_hours=("count", "size"), observed_dates=("date", "nunique"),
            total_rentals=("count", "sum"), user_rentals=(user, "sum"),
            mean_rentals=(user, "mean"), median_rentals=(user, "median"),
            std_rentals=(user, "std"), q25_rentals=(user, lambda s: s.quantile(0.25)),
            q75_rentals=(user, lambda s: s.quantile(0.75)),
        ).reset_index()
        part["user_type"] = user
        part["rental_share"] = part["user_rentals"] / part["total_rentals"].where(part["total_rentals"].gt(0))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_tables(data, pd):
    groups = {
        "overall": ["scope"], "hour": ["hour"], "weekday": ["weekday"],
        "workingday": ["workingday"], "weekend": ["is_weekend"],
        "holiday": ["holiday"], "workingday_hour": ["workingday", "hour"],
        "weekend_hour": ["is_weekend", "hour"], "weekday_hour": ["weekday", "hour"],
        "season": ["season"], "year_season": ["year", "season"],
        "year_month": ["year", "month", "year_month"], "year_hour": ["year", "hour"],
    }
    tables = {name: summarize(data, keys, pd) for name, keys in groups.items()}
    peaks = []
    for name in ["hour", "workingday_hour", "weekend_hour", "year_hour"]:
        table = tables[name]
        keys = ["user_type"] + [key for key in groups[name] if key != "hour"]
        maximum = table.groupby(keys)["mean_rentals"].transform("max")
        part = table.loc[table["mean_rentals"].eq(maximum)].copy()
        part.insert(0, "comparison", name)
        peaks.append(part)
    tables["peak_hours"] = pd.concat(peaks, ignore_index=True)
    return tables


# %% 七张图：英文图中文字；未观测组合留空
def save_figures(tables, directory, plt):
    directory.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"05_{name}.png", dpi=160)
        plt.close(fig)

    def curves(ax, table, x, full_range, metric):
        for user in USERS:
            part = table.loc[table["user_type"].eq(user)].set_index(x).reindex(full_range)
            ax.plot(part.index, part[metric], marker="o", label=user, color=COLORS[user])
        ax.set_xlabel(x)
        ax.set_ylabel("Share of rentals" if metric == "rental_share" else "Mean rentals per observed hour")
        if metric == "rental_share":
            ax.set_ylim(0, 1)
        ax.legend()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    overall = tables["overall"].set_index("user_type").reindex(USERS)
    axes[0].bar(USERS, overall["user_rentals"], color=[COLORS[u] for u in USERS])
    axes[0].set(title="Observed rental volume", ylabel="Rental events (not unique people)")
    axes[1].bar(USERS, overall["rental_share"], color=[COLORS[u] for u in USERS])
    axes[1].set(title="Overall rental composition", ylabel="Share of rentals", ylim=(0, 1))
    save(fig, "overall_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["mean_rentals", "rental_share"]):
        curves(ax, tables["hour"], "hour", range(24), metric)
        ax.set_title("Hourly user patterns")
    save(fig, "hourly_patterns")

    for name, key, labels in [
        ("workingday_hour", "workingday", {0: "Non-working day", 1: "Working day"}),
        ("weekend_hour", "is_weekend", {0: "Mon-Fri", 1: "Sat-Sun"}),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for column, value in enumerate([0, 1]):
            part = tables[name].loc[tables[name][key].eq(value)]
            for row, metric in enumerate(["mean_rentals", "rental_share"]):
                curves(axes[row, column], part, "hour", range(24), metric)
                axes[row, column].set_title(labels[value])
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["mean_rentals", "rental_share"]):
        curves(ax, tables["weekday"], "weekday", range(7), metric)
        ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_title("Weekday user patterns")
    save(fig, "weekday_patterns")

    for name, key, full_range in [("year_season", "season", range(1, 5)), ("year_month", "month", range(1, 13))]:
        years = sorted(tables[name]["year"].unique())
        fig, axes = plt.subplots(len(years), 2, figsize=(12, 4 * len(years)), squeeze=False)
        for row, year in enumerate(years):
            part = tables[name].loc[tables[name]["year"].eq(year)]
            for ax, metric in zip(axes[row], ["mean_rentals", "rental_share"]):
                curves(ax, part, key, full_range, metric)
                ax.set_xticks(list(full_range))
                ax.set_title(f"{year}: {key} (observed hours only)")
        save(fig, name)


def main():
    args = parse_args()
    if not args.input.is_file():
        print(f"缺少输入：{args.input}\n请先完成第01、02步。", file=sys.stderr)
        return 1
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"缺少依赖：{exc}\n请运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    try:
        data = load_data(args.input, pd, np)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"输入检查失败：{exc}\n本次未更新输出；旧文件不代表本次结果。", file=sys.stderr)
        return 1
    tables = build_tables(data, pd)
    notes = [
        ("unit", "分析单位为每小时租赁次数；用户类型不代表独立用户人数或出行目的。"),
        ("share", "rental_share=该组该类租赁量总和/该组count总和，不平均逐小时占比。"),
        ("zero_total", "组内总租赁量为0时占比为空；两类均值仍为0。未观测组合不补零。"),
        ("calendar", "工作日状态和周末分别比较；workingday=0不能直接称为周末。"),
        ("coverage", "训练集通常仅覆盖每月1–19日；总量为观测租赁次数，不代表完整月度总需求。"),
        ("time", "时间字段从datetime派生；weekday为周一0至周日6，season保留原始编码。"),
        ("quality", "保留第02步全部复核记录，不自动删除极端值或零值。"),
        ("peak", "每类用户按小时均值寻找峰值，保留全部并列最高小时；需结合样本数解释。"),
        ("inference", "组间差异仅为描述性结果；时间、季节和天气可能混杂，不作因果判断。"),
    ]
    tables["interpretation_notes"] = pd.DataFrame(notes, columns=["topic", "note"])
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.tables_dir / f"05_{name}.csv", index=False, encoding="utf-8-sig")
    save_figures(tables, args.figures_dir, plt)
    print(f"完成：{len(data)}个观测小时，{len(tables)}张表、7张图。")
    print(f"表格：{args.tables_dir.resolve()}\n图片：{args.figures_dir.resolve()}")
    print("租赁量占比不是用户人数占比；请结合各组观测小时数解释差异。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
