"""RQ4：分布统计、质量复核及统一分析数据生成。

不补零、不插值、不删除离群值。遇到缺失、非法值或重复时间戳时，
保存问题清单并退出，避免将有待处理的数据交给后续分析。
"""

# %% 路径与参数（不依赖当前工作目录）
import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "datetime", "season", "holiday", "workingday", "weather", "temp",
    "atemp", "humidity", "windspeed", "casual", "registered", "count",
]
DEMAND = ["count", "casual", "registered"]
WEATHER = ["temp", "atemp", "humidity", "windspeed"]
INTEGER_COLUMNS = ["season", "holiday", "workingday", "weather", *DEMAND]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/train.csv")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    return parser.parse_args()


# %% 读取与校验：将待处理问题定位到原始 CSV 记录
def validate(raw, pd, np):
    data = raw.copy()
    issues = []

    def flag(mask, field, issue):
        for index in raw.index[mask]:
            issues.append({"csv_row": int(index) + 2, "datetime": raw.loc[index, "datetime"],
                           "field": field, "issue": issue, "value": raw.loc[index, field]})

    for field in COLUMNS:
        missing = raw[field].isna() | raw[field].str.strip().eq("")
        if field == "datetime":
            data[field] = pd.to_datetime(raw[field], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        else:
            data[field] = pd.to_numeric(raw[field], errors="coerce")
        flag(missing, field, "missing")
        flag(~missing & data[field].isna(), field, "invalid_type")

    dates = data["datetime"]
    flag(dates.notna() & dates.duplicated(keep=False), "datetime", "duplicate_timestamp")
    flag(dates.notna() & dates.ne(dates.dt.floor("h")), "datetime", "off_hour_timestamp")
    allowed = {"season": [1, 2, 3, 4], "weather": [1, 2, 3, 4],
               "holiday": [0, 1], "workingday": [0, 1]}
    for field in COLUMNS[1:]:
        values = data[field]
        finite = values.notna() & np.isfinite(values)
        bad = values.notna() & ~finite
        bad |= finite & values.lt(0)
        if field in allowed:
            bad |= finite & ~values.isin(allowed[field])
        if field in INTEGER_COLUMNS:
            bad |= finite & values.mod(1).ne(0)
        if field == "humidity":
            bad |= finite & values.gt(100)
        flag(bad, field, "invalid_domain")
    complete = data[DEMAND].notna().all(axis=1)
    flag(complete & data["count"].ne(data["casual"] + data["registered"]),
         "count", "count_identity_mismatch")
    return data, pd.DataFrame(issues, columns=["csv_row", "datetime", "field", "issue", "value"])


# %% 描述统计、复核标记与时间字段
def prepare_data(data, pd, np):
    data = data.copy()
    # 保留源记录序号，之后按时间排序也能追溯到输入文件。
    data["source_csv_row"] = data.index + 2
    for field in INTEGER_COLUMNS:
        data[field] = data[field].astype("int64")
    time = data["datetime"].dt
    data["date"] = time.strftime("%Y-%m-%d")
    data["year"] = time.year
    data["month"] = time.month
    data["year_month"] = time.strftime("%Y-%m")
    data["day"] = time.day
    data["hour"] = time.hour
    data["weekday"] = time.dayofweek  # Monday=0, Sunday=6
    data["is_weekend"] = data["weekday"].ge(5).astype(int)
    data["flag_zero_humidity"] = data["humidity"].eq(0).astype(int)
    data["flag_zero_windspeed"] = data["windspeed"].eq(0).astype(int)
    data["log1p_count"] = np.log1p(data["count"])

    # 分母为零时占比未定义，保留为空，不生成无穷值。
    denominator = data["count"].where(data["count"].gt(0))
    data["registered_share"] = data["registered"] / denominator
    data["casual_share"] = data["casual"] / denominator
    thresholds = []
    for field in DEMAND + WEATHER:
        q1, q3 = data[field].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = data[field].lt(lower) | data[field].gt(upper)
        data[f"flag_iqr_{field}"] = mask.astype(int)
        thresholds.append(dict(field=field, q1=q1, q3=q3, iqr=iqr,
                               lower_fence=lower, upper_fence=upper, flagged_rows=int(mask.sum())))
    flag_columns = [column for column in data if column.startswith("flag_")]
    data["flag_any_review"] = data[flag_columns].any(axis=1).astype(int)
    # 标记不等于错误；极端需求和零天气值全部保留。
    return data.sort_values("datetime").reset_index(drop=True), pd.DataFrame(thresholds)


def save_plots(data, directory, plt, np):
    """采用英文图中文字，避免环境缺少中文字体导致方框。"""
    directory.mkdir(parents=True, exist_ok=True)
    for label, columns in [("demand", DEMAND), ("weather", WEATHER)]:
        fig, axes = plt.subplots(len(columns), 2, figsize=(11, 3 * len(columns)), squeeze=False)
        for row, field in enumerate(columns):
            axes[row, 0].hist(data[field], bins=40, color="#3577A8", edgecolor="white")
            axes[row, 0].set(xlabel=field, ylabel="Observed hours", title=f"Distribution: {field}")
            axes[row, 1].boxplot(data[field], vert=False)
            axes[row, 1].set(xlabel=field, title=f"Boxplot: {field} (flags only)")
        fig.tight_layout()
        fig.savefig(directory / f"02_{label}_distributions.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log1p(data["count"]), bins=40, color="#3577A8", edgecolor="white")
    ax.set(xlabel="log1p(count)", ylabel="Observed hours", title="Total rental count: log1p scale")
    fig.tight_layout()
    fig.savefig(directory / "02_log1p_count.png", dpi=160)
    plt.close(fig)


# %% 生成数据与输出表格
def main():
    args = parse_args()
    if not args.input.is_file():
        print(f"缺少输入文件：{args.input}\n请先放入 train.csv，并运行第 01 个审计脚本。", file=sys.stderr)
        return 1
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # 保存图片，不打开窗口；适用于终端运行。
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"缺少分析依赖：{exc}\n请先运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    # 防止自定义输出目录导致覆盖原始 train.csv。
    target = args.processed_dir / "train_eda.csv"
    if target.resolve() == args.input.resolve():
        print("输出文件不能与原始输入文件相同。", file=sys.stderr)
        return 1
    try:
        with args.input.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != COLUMNS:
            print(f"训练集表头不符。\n预期：{COLUMNS}\n实际：{header}", file=sys.stderr)
            return 1
        raw = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 1

    data, issues = validate(raw, pd, np)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    issues.to_csv(args.tables_dir / "02_validation_issues.csv", index=False, encoding="utf-8-sig")
    if raw.empty or not issues.empty:
        print(f"数据为空或存在 {len(issues)} 条质量问题，详见 02_validation_issues.csv。\n"
              "本次未生成分析数据或图表；如有旧输出，不代表本次结果。请复核后重新运行。", file=sys.stderr)
        return 1

    data, thresholds = prepare_data(data, pd, np)
    stats = data[DEMAND + WEATHER].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["skewness"] = data[DEMAND + WEATHER].skew()
    stats["zero_rows"] = data[DEMAND + WEATHER].eq(0).sum()
    stats.index.name = "field"
    stats.to_csv(args.tables_dir / "02_descriptive_statistics.csv", encoding="utf-8-sig")
    thresholds.to_csv(args.tables_dir / "02_iqr_thresholds.csv", index=False, encoding="utf-8-sig")
    data.loc[data["flag_any_review"].eq(1)].to_csv(
        args.tables_dir / "02_review_rows.csv", index=False, encoding="utf-8-sig")
    policies = [
        ("validation", "缺失、非法值、重复时间戳、非整点或总量不一致时停止，不自动修复。"),
        ("zero_weather", "零湿度、零风速仅标记，保留原值。"),
        ("iqr", "按全样本 1.5×IQR 标记七个连续变量；不是错误判定，不删除、不缩尾。"),
        ("missing_hours", "不补时间行、不补零、不插值；结合第01步时间覆盖表解释。"),
        ("time", "按时间排序；weekday 为周一0至周日6；周末独立于 workingday。"),
        ("share", "逐小时占比用于观察；分组占比应重新用分组租赁量总和相除，不直接平均。"),
        ("zero_count", "count=0 时两类用户占比为空，这是派生值未定义，不是原始缺失。"),
        ("labels", "season、weather 保留原始编码，未验证单位和编码解释前不添加标签。"),
    ]
    pd.DataFrame(policies, columns=["topic", "decision"]).to_csv(
        args.tables_dir / "02_processing_policy.csv", index=False, encoding="utf-8-sig")
    save_plots(data, args.figures_dir, plt, np)
    # 最后写统一分析数据，只有完整流程走通才更新该文件。
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    print(f"完成：保留全部 {len(data)} 行，其中 {int(data['flag_any_review'].sum())} 行需复核。")
    print(f"分析数据：{target.resolve()}")
    print(f"统计表：{args.tables_dir.resolve()}；三张图：{args.figures_dir.resolve()}")
    print("IQR 标记与天气零值仅用于复核，不能直接解释为数据错误。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
