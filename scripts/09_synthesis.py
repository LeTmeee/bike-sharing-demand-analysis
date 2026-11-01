"""第09步：汇总第01–08步的审计、EDA、推断、聚类与特征选择证据。

基础证据重新计算；扩展表必须与本次输入 SHA256 一致，缺失或不匹配即停止。
输出 reports/synthesis_summary.md，不覆盖手工编写的 final_report.md。
"""

# %% 参数与共享函数
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WEATHER = ["temp", "atemp", "humidity", "windspeed"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train_eda.csv")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--source-tables-dir", type=Path, default=ROOT / "outputs/tables",
                        help="第06–08步结果与元数据目录；可以与本次输出目录不同")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw",
                        help="第01步的三个原始CSV，train须与本次处理后输入对应")
    parser.add_argument("--min-samples", type=int, default=30)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_upstream(input_path, directory, pd):
    """Read existing runs without refitting or using final test scores to tune.

    The input hash checks data provenance, not the statistical validity of a run.
    Source table hashes are captured for a reviewable record of this synthesis.
    """
    required = {
        '06': {'tests': ['analysis', 'test', 'n', 'p_value', 'p_holm_three_tests', 'effect', 'estimate', 'ci_low', 'ci_high'],
               'weather_posthoc': ['analysis', 'weather_a', 'weather_b', 'p_holm']},
        '07': {'cluster_summary': ['representation', 'cluster', 'days', 'workingday_share', 'peak_hours'],
               'k_metrics': ['representation', 'k', 'silhouette', 'selected'],
               'retention': ['retained_fraction']},
        '08': {'comparison': ['feature', 'lasso_selected', 'rmse_increase', 'repeat_sd', 'spearman_rho'],
               'model_metrics': ['model', 'split', 'rmse', 'mae', 'r2'],
               'spearman': ['feature', 'temp', 'atemp', 'count']},
    }
    tables, runs, sources = {}, {}, []
    digest = sha256(input_path)
    for step, names in required.items():
        path = directory / f'{step}_run_metadata.json'
        if not path.is_file():
            raise ValueError(f'缺少第{step}步元数据：{path}；请先运行对应脚本')
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('input_sha256') != digest:
            raise ValueError(f'第{step}步 SHA256 与当前输入不一致；请重新运行该步骤')
        runs[step] = meta
        sources.append(dict(step=step, path=str(path.resolve()), sha256=sha256(path)))
        for name, columns in names.items():
            path = directory / f'{step}_{name}.csv'
            if not path.is_file():
                raise ValueError(f'缺少证据表：{path}；请重新运行第{step}步')
            table = pd.read_csv(path, encoding='utf-8-sig')
            if not set(columns).issubset(table.columns) or (table.empty and name != 'weather_posthoc'):
                raise ValueError(f'证据表为空或字段不兼容：{path}')
            tables[f'{step}_{name}'] = table
            sources.append(dict(step=step, path=str(path.resolve()), sha256=sha256(path)))
    return tables, runs, sources


def audit_raw(directory, data, pd, np):
    """Reuse step 01's actual checks on current raw files, never old audit tables."""
    spec = importlib.util.spec_from_file_location('audit_helpers', Path(__file__).with_name('01_data_audit.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks, fields, categories, flagged = [], [], [], []
    dates, frames, sources = {}, {}, []
    for name, expected in module.SCHEMAS.items():
        path = directory / name
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        if list(frame) != expected:
            raise ValueError(f'原始文件字段不兼容：{path}')
        dates[name] = module.audit_file(name, frame, pd, checks, fields, categories, flagged)
        frames[name] = frame
        sources.append(dict(step='01', path=str(path.resolve()), sha256=sha256(path)))
    overlap = len(set(dates['train.csv'].dropna()) & set(dates['test.csv'].dropna()))
    aligned = dates['test.csv'].reset_index(drop=True).equals(dates['sampleSubmission.csv'].reset_index(drop=True))
    checks.extend([
        dict(file='train+test', check='timestamp_overlap', value=overlap, status='FAIL' if overlap else 'PASS', note=''),
        dict(file='test+sample', check='timestamp_order_identical', value=aligned, status='PASS' if aligned else 'FAIL', note=''),
    ])
    result = pd.DataFrame(checks)
    if result.status.eq('FAIL').any():
        raise ValueError('原始数据审计存在 FAIL；请先运行01_data_audit.py检查')
    raw = frames['train.csv'].copy()
    raw['datetime'] = pd.to_datetime(raw.datetime)
    raw = raw.sort_values('datetime').reset_index(drop=True)
    current = data.sort_values('datetime').reset_index(drop=True)
    if len(raw) != len(current) or not raw.datetime.equals(current.datetime):
        raise ValueError('原始train与处理后输入时间范围不匹配；核对 --raw-dir 和 --input')
    for col in module.SCHEMAS['train.csv'][1:]:
        if not np.allclose(pd.to_numeric(raw[col]), current[col], rtol=1e-10, atol=1e-10):
            raise ValueError(f'原始train与处理后输入字段不一致：{col}')
    return result, sources


def load_user_helpers():
    path = Path(__file__).with_name("05_user_behavior.py")
    spec = importlib.util.spec_from_file_location("user_behavior_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(path, helpers, pd, np):
    data = helpers.load_data(path, pd, np)
    for field in ["weather", *WEATHER]:
        if field not in data:
            raise ValueError(f"缺少字段：{field}")
        values = pd.to_numeric(data[field], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"{field} 存在缺失或非法数值。")
        if field == "weather" and not values.isin([1, 2, 3, 4]).all():
            raise ValueError("weather 类别非法。")
        if field == "humidity" and values.gt(100).any():
            raise ValueError("humidity 超过100。")
        data[field] = values
    data["weather"] = data["weather"].astype(int)
    return data


# %% 关键表与组合分析
def build_tables(data, helpers, pd, np, minimum):
    tables = {}
    groups = {
        "overall_users": ["scope"], "workingday_hour_users": ["workingday", "hour"],
        "weekday_users": ["weekday"], "year_month_users": ["year", "month"],
        "year_season_users": ["year", "season"],
        "weather_users": ["weather"],
        "weather_workingday_hour_users": ["workingday", "weather", "hour"],
    }
    for name, keys in groups.items():
        table = helpers.summarize(data, keys, pd)
        table["sparse_group"] = table["observed_hours"].lt(minimum).astype(int)
        tables[name] = table

    peak_table = tables["workingday_hour_users"]
    maximum = peak_table.groupby(["workingday", "user_type"])["mean_rentals"].transform("max")
    tables["peak_hours"] = peak_table.loc[peak_table["mean_rentals"].eq(maximum)].copy()
    correlations = []
    for field in WEATHER:
        x, y = data[field], data["count"]
        valid = len(data) >= 2 and x.nunique() > 1 and y.nunique() > 1
        correlations.append(dict(field=field, observed_hours=len(data),
                                 pearson=x.corr(y) if valid else np.nan,
                                 spearman=x.rank().corr(y.rank()) if valid else np.nan))
    tables["weather_correlations"] = pd.DataFrame(correlations)

    # 敏感性分析仅对副本筛选；主分析保留全样本。
    q1, q3 = data["count"].quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    subsets = {
        "all": data,
        "count_within_iqr_fences": data.loc[data["count"].between(lower, upper)],
        "positive_humidity": data.loc[data["humidity"].gt(0)],
        "positive_windspeed": data.loc[data["windspeed"].gt(0)],
    }
    sensitivity = []
    for label, part in subsets.items():
        total = part["count"].sum()
        sensitivity.append(dict(subset=label, observed_hours=len(part),
                                excluded_hours=len(data) - len(part), mean_count=part["count"].mean(),
                                median_count=part["count"].median(),
                                registered_share=part["registered"].sum() / total if total else np.nan))
    tables["sensitivity"] = pd.DataFrame(sensitivity)
    tables["quality_overview"] = pd.DataFrame([
        ("observed_hours", len(data)), ("observed_dates", data["date"].nunique()),
        ("mean_count", data["count"].mean()), ("median_count", data["count"].median()),
        ("skewness_count", data["count"].skew()), ("maximum_count", data["count"].max()),
        ("zero_humidity_rows", int(data["humidity"].eq(0).sum())),
        ("zero_windspeed_rows", int(data["windspeed"].eq(0).sum())),
        ("count_iqr_lower_fence", lower), ("count_iqr_upper_fence", upper),
        ("count_iqr_flagged_rows", int((~data["count"].between(lower, upper)).sum())),
    ], columns=["metric", "value"])
    # Recompute source-variable EDA summaries; do not average group averages.
    for name, keys in [('hourly_demand', ['hour']), ('seasonal_demand', ['year', 'season']),
                       ('weather_demand', ['weather'])]:
        tables[name] = data.groupby(keys)['count'].agg(
            observed_hours='size', mean_count='mean', median_count='median').reset_index()
    return tables


# %% 图表：组合条件下的均值、构成和样本量
def save_figures(tables, directory, plt):
    directory.mkdir(parents=True, exist_ok=True)
    combo = tables["weather_workingday_hour_users"]
    # 每类用户共享总需求和小时样本数，取其中一类，避免重复计算。
    total = combo.loc[combo["user_type"].eq("registered")].copy()
    total["mean_count"] = total["total_rentals"] / total["observed_hours"]

    def heatmap(matrix, ax, maximum):
        # Matplotlib handles missing cells without filling them with zero.
        im = ax.imshow(matrix.to_numpy(dtype=float), aspect='auto', cmap='YlGnBu', vmin=0, vmax=maximum)
        ax.set_xticks(range(24), range(24))
        ax.set_yticks(range(4), [1, 2, 3, 4])
        ax.figure.colorbar(im, ax=ax)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"09_{name}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for column, workingday in enumerate([0, 1]):
        part = total.loc[total["workingday"].eq(workingday)]
        for row, metric in enumerate(["mean_count", "observed_hours"]):
            matrix = part.pivot(index="weather", columns="hour", values=metric).reindex(index=[1, 2, 3, 4], columns=range(24))
            heatmap(matrix, axes[row, column], total[metric].max())
            axes[row, column].set(title=f"workingday={workingday}: {metric}", ylabel="Weather code", xlabel="Hour")
    save(fig, "weather_workingday_hour")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, workingday in zip(axes, [0, 1]):
        part = total.loc[total["workingday"].eq(workingday)]
        matrix = part.pivot(index="weather", columns="hour", values="rental_share").reindex(index=[1, 2, 3, 4], columns=range(24))
        heatmap(matrix, ax, 1)
        ax.set(title=f"Registered rental share: workingday={workingday}", xlabel="Hour", ylabel="Weather code")
    save(fig, "registered_share_conditions")

    sensitivity = tables["sensitivity"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(sensitivity)), sensitivity["mean_count"])
    ax.set_xticks(range(len(sensitivity)), sensitivity["subset"], rotation=15, ha="right")
    ax.set(ylabel="Mean rentals per observed hour", title="Sensitivity comparisons (different subsets)")
    for index, row in sensitivity.iterrows():
        if row["observed_hours"]:
            ax.annotate(f"n={int(row['observed_hours'])}", (index, row["mean_count"]), ha="center")
    save(fig, "sensitivity")


def extension_evidence(upstream, runs, directory):
    """Preserve analysis units and distinguish screening from model performance."""
    evidence = []

    def add(step, finding, name, limitation):
        evidence.append(dict(research_question=step, finding=finding,
                             evidence_file=str((directory / name).resolve()), limitation=limitation))

    tests = upstream['06_tests']
    primary = tests.loc[tests.analysis.eq('complete_days_primary')]
    if len(primary) != 3:
        raise ValueError('第06步缺少三个完整日期主检验')
    for row in primary.itertuples():
        ci = f'；按日期bootstrap区间 [{row.ci_low:.4f}, {row.ci_high:.4f}]' if row.ci_low == row.ci_low else ''
        interpretation = '存在拒绝原假设的统计证据' if row.p_holm_three_tests < .05 else '没有足够证据拒绝原假设'
        add('06', f'{row.test}：n={row.n}，原始p={row.p_value:.4g}，三项Holm校正p={row.p_holm_three_tests:.4g}，'
            f'{row.effect}={row.estimate:.4f}{ci}；以0.05为阈值，{interpretation}。', '06_tests.csv',
            '以日期为分析单位；Mann–Whitney检验不是均值相等检验。跨日独立仍是假设，未控制季节、年份混杂。')
    sensitivity = tests.loc[tests.analysis.eq('all_days_sensitivity')]
    for row in sensitivity.itertuples():
        add('06', f'全部日期敏感性：{row.test}，n={row.n}，Holm校正p={row.p_holm_three_tests:.4g}，{row.effect}={row.estimate:.4f}。',
            '06_tests.csv', '不完整日期的小时构成不同；敏感性结果不替代完整日主分析。')
    post = upstream['06_weather_posthoc']
    for row in post.loc[post.analysis.eq('complete_days_primary')].itertuples():
        add('06', f'天气事后比较 {row.weather_a} vs {row.weather_b}：组内Holm校正p={row.p_holm:.4g}。',
            '06_weather_posthoc.csv', '天气编码为每日主要天气；排除主要天气并列日和过少样本类别，详见第06步。')

    clusters = upstream['07_cluster_summary']
    for representation, choice in runs['07']['selected'].items():
        part = clusters.loc[clusters.representation.eq(representation)]
        add('07', f'{representation} 表示选择 K={choice["k"]}，silhouette={choice["silhouette"]:.4f}，'
            f'使用 {int(part.days.sum())} 个日期。', '07_k_metrics.csv',
            'share比较日内形状，raw比较规模与形状；K由探索性silhouette选择，不是已知真实类别。')
        for row in part.itertuples():
            add('07', f'{representation} 群{row.cluster}：{row.days}日，工作日占比{row.workingday_share:.2%}，'
                f'平均曲线峰值小时={row.peak_hours}。', '07_cluster_summary.csv',
                '群标签不表示出行目的；工作日与天气仅作事后解释，未参与聚类拟合。')
    cm = runs['07']
    add('07', f'观测{cm["observed_days"]}日，保留{cm["included_days"]}日；'
        f'不完整{cm["incomplete_days"]}日，时间范围内全缺失{cm["absent_dates"]}日。',
        '07_retention.csv', '只对完整24小时且正总量日期聚类，不补零；筛选可能改变季节构成。')

    comp = upstream['08_comparison']
    selected = comp.lasso_selected.astype(str).str.lower().eq('true')
    add('08', f'Lasso alpha={runs["08"]["best_alpha"]:.6g}，保留{int(selected.sum())}/{len(comp)}个源变量：'
        + '、'.join(comp.loc[selected, 'feature']) + '。', '08_comparison.csv',
        '每折内重新标准化与one-hot编码；相关变量可能互相替代，非零系数不是显著性结论。')
    for row in comp.nlargest(3, 'rmse_increase').itertuples():
        add('08', f'训练期验证置换：{row.feature}，RMSE增加{row.rmse_increase:.3f} ± {row.repeat_sd:.3f}。',
            '08_comparison.csv', '±为重复置换标准差，不是置信区间；逐字段打乱可能形成不自然的组合。')
    for row in comp.dropna(subset=['spearman_rho']).itertuples():
        add('08', f'训练期探索性Spearman：{row.feature} 与count，rho={row.spearman_rho:.4f}。',
            '08_comparison.csv', '与第04步全样本或第06步每日聚合使用不同样本/单位；不据此自动筛选或解释因果。')
    for row in upstream['08_model_metrics'].itertuples():
        add('08', f'{row.model} / {row.split}：RMSE={row.rmse:.3f}，MAE={row.mae:.3f}，R²={row.r2:.4f}。',
            '08_model_metrics.csv', '测试期独立保留，未用于调参、排名或筛选；没有比较缩减变量模型，不能声称筛选提高了精度。')
    fm = runs['08']
    add('08', f'训练{fm["train_hours"]}小时（{fm["train_start"]}—{fm["train_end"]}）；'
        f'最终测试{fm["test_hours"]}小时（{fm["test_start"]}—{fm["test_end"]}）。',
        '08_run_metadata.json', '排除casual、registered、用户占比、log1p_count和标记字段；按日期先后划分，天气为同时段观测值。')
    return evidence


# %% 自动中文证据摘要，不预设研究结论
def write_report(data, tables, args, pd, extension, metadata):
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    target = args.reports_dir / "synthesis_summary.md"

    def link(path):
        try:
            return Path(os.path.relpath(path.resolve(), target.parent.resolve())).as_posix()
        except ValueError:
            # Windows cannot express a relative path between different drives.
            return path.resolve().as_posix()

    evidence = []

    def add(rq, finding, source, limitation):
        evidence.append(dict(research_question=rq, finding=finding, evidence_file=source, limitation=limitation))

    audit = tables['raw_audit_checks']
    add('01', f'重新审计三个原始文件：FAIL={int(audit.status.eq("FAIL").sum())}项，'
        f'WARN={int(audit.status.eq("WARN").sum())}项；原始train的12个字段与当前处理后输入一致。',
        '09_raw_audit_checks.csv', 'WARN表示待复核而非错误；test.csv和sampleSubmission.csv不提供真实需求标签。')
    add("RQ4", f"观测小时数 {len(data):,}；总需求均值 {data['count'].mean():.2f}、中位数 {data['count'].median():.2f}、偏度 {data['count'].skew():.2f}。",
        "09_quality_overview.csv", "描述的是处理后训练数据；不能据此证明原始文件没有缺失或错误。")
    for _, row in tables["peak_hours"].iterrows():
        add("RQ1", f"workingday={int(row['workingday'])} 时，{row['user_type']} 的最高小时均值位于 {int(row['hour']):02d}:00，为 {row['mean_rentals']:.2f} 次（n={int(row['observed_hours'])}）。",
            "09_peak_hours.csv", "按各用户类型分别计算，保留并列峰值；不代表该时段每一天都是最高。")
    for _, row in tables["weather_correlations"].iterrows():
        add("RQ2", f"{row['field']} 与 count：Pearson={row['pearson']:.3f}，Spearman={row['spearman']:.3f}。",
            "09_weather_correlations.csv", "相关性不是因果效应；nan 表示相关系数未定义。")
    for _, row in tables["overall_users"].iterrows():
        share = f"{row['rental_share']:.2%}" if pd.notna(row["rental_share"]) else "未定义"
        add("RQ3", f"{row['user_type']}：观测租赁次数 {int(row['user_rentals']):,}，占比 {share}，小时均值 {row['mean_rentals']:.2f}。",
            "09_overall_users.csv", "占比为租赁次数占比，不是独立用户占比，也不能确定出行目的。")
    hourly = tables['hourly_demand']
    for row in hourly.loc[hourly.mean_count.eq(hourly.mean_count.max())].itertuples():
        add('RQ1', f'总需求最高小时均值：{int(row.hour):02d}:00，{row.mean_count:.2f}次，n={row.observed_hours}。',
            '09_hourly_demand.csv', '按实际观测小时计算；保留并列峰值，不代表每一天的峰值。')
    for row in tables['seasonal_demand'].itertuples():
        add('RQ1', f'{row.year}年，season={row.season}：小时均值{row.mean_count:.2f}，n={row.observed_hours}。',
            '09_seasonal_demand.csv', '保留原始季节编码；跨年比较受天气和样本构成影响。')
    for row in tables['weather_demand'].itertuples():
        add('RQ2', f'weather={row.weather}：小时均值{row.mean_count:.2f}，中位数{row.median_count:.2f}，n={row.observed_hours}。',
            '09_weather_demand.csv', '小样本天气类别不可形成一般性结论，日历构成可能混杂。')
    evidence.extend(extension)
    evidence_table = pd.DataFrame(evidence)
    evidence_table.to_csv(args.tables_dir / "09_evidence_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# Bike Sharing Demand 综合分析摘要", "",
             "> 此文件由第09步自动生成，再次运行会覆盖；最终课程报告请另行整理为 final_report.md。", "",
             f"分析输入：`{args.input.resolve()}`。", "",
             f"观测范围：{data['datetime'].min()} 至 {data['datetime'].max()}；仅统计实际观测小时。", "",
             f"处理后输入 SHA256：`{metadata['input_sha256']}`。", "",
             "本摘要综合前八项工作：第01步复用审计函数重新检查原始文件；第02–05步核心EDA从当前输入重算；"
             "第06–08步引用已运行的结果，并逐项校验输入SHA256一致。所有来源文件哈希保存在09_run_metadata.json中。"
             "哈希核对证明输入一致，不替代统计方法与代码复核。", ""]
    sections = [('01', '第01步：数据审计'), ('RQ4', '第02步：分布与质量（RQ4）'),
                ('RQ1', '第03步：时间模式（RQ1）'), ('RQ2', '第04步：天气关联（RQ2）'),
                ('RQ3', '第05步：用户差异（RQ3）'), ('06', '第06步：假设检验'),
                ('07', '第07步：聚类分析'), ('08', '第08步：特征选择与预测评估')]
    for rq, title in sections:
        lines.extend([f"## {title}", ""])
        for item in evidence:
            if item["research_question"] == rq:
                source = link(args.tables_dir / item["evidence_file"])
                lines.extend([f"- {item['finding']} [统计表]({source})。{item['limitation']}", ""])
    lines.extend(["## 组合分析与敏感性", "",
                  f"天气×工作日×小时汇总中的 sparse_group 表示 n<{args.min_samples}，仅是稀疏提示，不是显著性标准。", "",
                  "敏感性比较分别剔除 count 的 IQR 范围外记录、零湿度记录、零风速记录；这些子集只用于比较，不替代主分析。", "",
                  "样本筛选会改变时间和天气构成，均值变化不能证明被筛除记录错误。", ""])
    for name in ["weather_workingday_hour", "registered_share_conditions", "sensitivity"]:
        lines.extend([f"![{name}]({link(args.figures_dir / f'09_{name}.png')})", ""])
    lines.extend(["## 运营讨论与局限", "",
                  "- 时间曲线、用户峰值、聚类形状和模型中的小时重要性可交叉参照，作为排班调查依据；这些证据使用不同样本和方法，不是相互独立的因果验证。",
                  "- 工作日整体分布检验与小时交互预测价值回答不同问题；整体检验未拒绝原假设，不表示工作日信息没有预测价值。",
                  "- 先分别解释全小时天气关联、每日聚合检验和训练期模型关系，再综合讨论，不能把不同层次的相关系数直接当作矛盾或因果效应。",
                  "- 可将峰值小时作为后续车辆配置、补给与排班调查的候选时段；数据不含站点库存，不能直接给出调拨数量。",
                  "- 对天气相关的需求差异，应同时检查小时、工作日和样本量；极少天气样本不能形成一般性运营结论。",
                  "- 用户类别差异可帮助描述需求构成，但不等于已经识别出通勤或休闲动机。",
                  "- 训练集通常仅含每月1–19日；月度、季节汇总不是完整期间需求，缺失小时不补零。",
                  "- 年月与季节证据分别见09_year_month_users.csv和09_year_season_users.csv，需结合年份和样本数进一步解释。",
                  "- 结果限于该系统和观测时期；不作因果推断或长期趋势外推。", "", "## 本次输出索引", ""])
    for name in [*tables, "evidence_summary"]:
        filename = f"09_{name}.csv"
        lines.append(f"- [{filename}]({link(args.tables_dir / filename)})")
    lines.extend(['', '## 复现', '', '```powershell',
                  'python scripts/09_synthesis.py',
                  "python -m unittest discover -s scripts -p 'test_*.py'", '```', '',
                  '先按README运行第01–08步。--source-tables-dir指定06–08证据目录；--raw-dir指定原始文件目录；'
                  '--input指定处理后输入；--tables-dir、--figures-dir、--reports-dir指定本次产物目录。'
                  '脚本缺文件、哈希不一致或校验失败时停止；旧报告仍可能存在，请核对本次退出码。', '',
                  f'[运行元数据]({link(args.tables_dir / "09_run_metadata.json")})'])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main():
    args = parse_args()
    if args.min_samples < 1 or not args.input.is_file():
        print("输入文件不存在或 --min-samples 不是正整数；请先完成第01–05步。", file=sys.stderr)
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
        helpers = load_user_helpers()
        data = load_data(args.input, helpers, pd, np)
        upstream, runs, sources = load_upstream(args.input, args.source_tables_dir, pd)
        extension = extension_evidence(upstream, runs, args.source_tables_dir)
        raw_checks, raw_sources = audit_raw(args.raw_dir, data, pd, np)
        tables = build_tables(data, helpers, pd, np, args.min_samples)
        tables['raw_audit_checks'] = raw_checks
        source_paths = [args.input, Path(__file__), Path(__file__).with_name('01_data_audit.py'),
                        Path(__file__).with_name('05_user_behavior.py')]
        metadata = dict(input_sha256=sha256(args.input), input=str(args.input.resolve()),
                        upstream_runs=runs, min_samples=args.min_samples,
                        versions=dict(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__),
                        evidence_files=sources + raw_sources + [
                            dict(step='09', path=str(p.resolve()), sha256=sha256(p)) for p in source_paths],
                        policy='01 raw audit and 02-05 core evidence recomputed; 06-08 input hashes verified; no retuning')
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        print(f"输入检查失败：{exc}\n本次未更新输出；旧文件不代表本次结果。", file=sys.stderr)
        return 1
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.tables_dir / f"09_{name}.csv", index=False, encoding="utf-8-sig")
    save_figures(tables, args.figures_dir, plt)
    report = write_report(data, tables, args, pd, extension, metadata)
    (args.tables_dir / '09_run_metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"完成：{len(data)}个观测小时，{len(tables) + 1}张表、3张图。\n摘要：{report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
