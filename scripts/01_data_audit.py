"""核验 Bike Sharing Demand 原始数据，不执行清洗或补值。

默认从项目的 data/raw 读取文件，向 outputs/tables 写入 audit_* 表格。
可在任何工作目录运行；VS Code 中也可使用 # %% 分段查看。
"""

# %% 路径、字段与命令行参数
from pathlib import Path
import argparse
import csv
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "datetime", "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
]
SCHEMAS = {
    "train.csv": FEATURES + ["casual", "registered", "count"],
    "test.csv": FEATURES,
    "sampleSubmission.csv": ["datetime", "count"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/tables")
    return parser.parse_args()


# %% 单文件审计
def audit_file(name, frame, pd, checks, fields, categories, flagged):
    """返回转换后的数值与时间，转换只用于检查，不覆盖输入。"""
    def check(item, value, status="INFO", note=""):
        checks.append(dict(file=name, check=item, value=value, status=status, note=note))

    check("row_count", len(frame), "PASS" if len(frame) else "FAIL")
    check("column_count", len(frame.columns))
    check("duplicate_rows", int(frame.duplicated().sum()),
          "WARN" if frame.duplicated().any() else "PASS")

    numeric = {}
    for column in frame.columns:
        raw = frame[column]
        missing = raw.isna() | raw.astype(str).str.strip().eq("")
        if column == "datetime":
            converted = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            dtype = "datetime"
        else:
            converted = pd.to_numeric(raw, errors="coerce")
            dtype = "numeric"
            numeric[column] = converted
        invalid = converted.isna() & ~missing
        fields.append(dict(
            file=name, field=column, expected_type=dtype,
            missing_count=int(missing.sum()), invalid_count=int(invalid.sum()),
            minimum=str(converted.min()), maximum=str(converted.max()),
            zero_count=int(converted.eq(0).sum()) if dtype == "numeric" else "",
        ))
        for label, mask in [("missing", missing), ("invalid_type", invalid)]:
            check(f"{column}:{label}", int(mask.sum()), "FAIL" if mask.any() else "PASS")
            for index in frame.index[mask]:
                flagged.append(dict(file=name, csv_row=int(index) + 2,
                                    field=column, issue=label, value=str(raw.loc[index])))

    dates = pd.to_datetime(frame["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    valid_dates = dates.dropna()
    duplicates = dates.notna() & dates.duplicated(keep=False)
    check("duplicate_timestamp_rows", int(duplicates.sum()), "FAIL" if duplicates.any() else "PASS")
    check("timestamps_sorted", bool(valid_dates.is_monotonic_increasing),
          "PASS" if valid_dates.is_monotonic_increasing else "WARN")
    off_hour = dates.notna() & dates.ne(dates.dt.floor("h"))
    check("off_hour_timestamp_rows", int(off_hour.sum()), "FAIL" if off_hour.any() else "PASS")
    for label, mask in [("duplicate_timestamp", duplicates), ("off_hour_timestamp", off_hour)]:
        for index in frame.index[mask]:
            flagged.append(dict(file=name, csv_row=int(index) + 2, field="datetime",
                                issue=label, value=str(frame.loc[index, "datetime"])))

    allowed = {"season": [1, 2, 3, 4], "weather": [1, 2, 3, 4],
               "holiday": [0, 1], "workingday": [0, 1]}
    for column, values in numeric.items():
        finite = values.notna() & ~values.isin([float("inf"), -float("inf")])
        bad = values.notna() & ~finite
        bad |= finite & values.lt(0)
        if column in allowed:
            bad |= finite & ~values.isin(allowed[column])
            for value, count in frame[column].value_counts(dropna=False).items():
                categories.append(dict(file=name, field=column, value=str(value), rows=int(count)))
        if column == "humidity":
            bad |= finite & values.gt(100)
        if column in ["casual", "registered", "count"]:
            bad |= finite & values.mod(1).ne(0)
        check(f"{column}:invalid_domain", int(bad.sum()), "FAIL" if bad.any() else "PASS",
              "检查有限值、非负、类别编码、湿度上限及计数整数；温度未设任意上限。")
        for index in frame.index[bad]:
            flagged.append(dict(file=name, csv_row=int(index) + 2, field=column,
                                issue="invalid_domain", value=str(frame.loc[index, column])))
        if column in ["humidity", "windspeed"]:
            zero = values.eq(0)
            check(f"{column}:zero_review", int(zero.sum()), "WARN" if zero.any() else "PASS",
                  "仅标记，零值不自动判错或替换。")
            for index in frame.index[zero]:
                flagged.append(dict(file=name, csv_row=int(index) + 2, field=column,
                                    issue="zero_review", value="0"))

    if name == "train.csv":
        complete = pd.DataFrame({k: numeric[k] for k in ["casual", "registered", "count"]}).notna().all(axis=1)
        mismatch = complete & numeric["count"].ne(numeric["casual"] + numeric["registered"])
        check("count_identity_uncheckable", int((~complete).sum()), "FAIL" if (~complete).any() else "PASS")
        check("count_identity_mismatch", int(mismatch.sum()), "FAIL" if mismatch.any() else "PASS")
        for index in frame.index[mismatch]:
            flagged.append(dict(file=name, csv_row=int(index) + 2, field="count",
                                issue="count_identity_mismatch", value=str(frame.loc[index, "count"])))
    if name == "sampleSubmission.csv":
        check("placeholder_count_all_zero", bool(numeric["count"].eq(0).all()),
              note="提交模板中的 count 不是真实需求。")
    return dates


# %% 主流程：先核验文件和表头，再执行检查、保存结果
def main():
    args = parse_args()
    missing = [str(args.data_dir / name) for name in SCHEMAS if not (args.data_dir / name).is_file()]
    if missing:
        print("缺少输入文件，请先放入以下文件：\n" + "\n".join(missing), file=sys.stderr)
        return 1
    try:
        import pandas as pd
    except ImportError:
        print("缺少 pandas，请先运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    frames = {}
    try:
        for name, expected in SCHEMAS.items():
            path = args.data_dir / name
            with path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            if header != expected:
                print(f"{name} 表头与预期不一致。\n预期：{expected}\n实际：{header}", file=sys.stderr)
                return 1
            # 保留原始文本以区分空字段和不可解析值；不将字符串 NA 等静默转为空值。
            frames[name] = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"读取 CSV 失败：{exc}", file=sys.stderr)
        return 1

    checks, fields, categories, flagged = [], [], [], []
    dates = {name: audit_file(name, frame, pd, checks, fields, categories, flagged)
             for name, frame in frames.items()}
    coverage, gaps = [], []
    for name, series in {**dates, "train+test": pd.concat([dates["train.csv"], dates["test.csv"]])}.items():
        valid = pd.DatetimeIndex(series.dropna().unique()).sort_values()
        if valid.empty:
            coverage.append(dict(file=name, first="", last="", observed_timestamps=0,
                                 expected_hours="", absent_hours="", day_numbers=""))
            continue
        grid = pd.date_range(valid.min().floor("h"), valid.max().ceil("h"), freq="h")
        absent = grid.difference(valid)
        coverage.append(dict(file=name, first=str(valid.min()), last=str(valid.max()),
                             observed_timestamps=len(valid), expected_hours=len(grid),
                             absent_hours=len(absent), day_numbers=",".join(map(str, sorted(set(valid.day))))))
        for stamp in absent:
            gaps.append(dict(file=name, datetime=str(stamp)))

    overlap = len(set(dates["train.csv"].dropna()) & set(dates["test.csv"].dropna()))
    checks.append(dict(file="train+test", check="timestamp_overlap", value=overlap,
                       status="FAIL" if overlap else "PASS", note="训练集与测试集应无时间交集。"))
    test_dates, sample_dates = dates["test.csv"], dates["sampleSubmission.csv"]
    aligned = (test_dates.notna().all() and sample_dates.notna().all()
               and test_dates.reset_index(drop=True).equals(sample_dates.reset_index(drop=True)))
    checks.append(dict(file="test+sample", check="timestamp_order_identical", value=bool(aligned),
                       status="PASS" if aligned else "FAIL", note="同时核对行数、时间值及顺序。"))

    # 所有审计表使用固定列，即使没有问题，明细表仍保留表头。
    tables = {
        "checks": (checks, ["file", "check", "value", "status", "note"]),
        "fields": (fields, ["file", "field", "expected_type", "missing_count", "invalid_count", "minimum", "maximum", "zero_count"]),
        "categories": (categories, ["file", "field", "value", "rows"]),
        "flagged_rows": (flagged, ["file", "csv_row", "field", "issue", "value"]),
        "time_coverage": (coverage, ["file", "first", "last", "observed_timestamps", "expected_hours", "absent_hours", "day_numbers"]),
        "absent_hours": (gaps, ["file", "datetime"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, (records, columns) in tables.items():
        pd.DataFrame(records, columns=columns).to_csv(
            args.output_dir / f"audit_{suffix}.csv", index=False, encoding="utf-8-sig")
    failures = sum(row["status"] == "FAIL" for row in checks)
    warnings = sum(row["status"] == "WARN" for row in checks)
    print(f"审计完成：{failures} 项失败，{warnings} 项待复核；输出：{args.output_dir.resolve()}")
    print("缺失小时按各文件最早至最晚时间计算；训练集包含划分给测试集的小时。")
    print("csv_row 为含表头的 CSV 记录序号；原始数据未修改，WARN 不代表数据错误。")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
