"""Reproducible feature selection extension (step 08). See reports/feature_selection.md.

Run from any directory. Input is never modified. Only allowlisted predictors
enter models. All Lasso preprocessing is refitted inside chronological CV.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS = ["temp", "atemp", "humidity", "windspeed"]
NUMERIC = CONTINUOUS + ["year"]
CATEGORICAL = ["hour", "weekday", "month", "season", "holiday", "workingday", "weather"]
ALPHAS = np.logspace(-2, 2, 25)
TOLERANCE = 1e-8


def prepare(data):
    """Validate required observed data; rederive time; discard every extra column."""
    required = ["datetime", "count", *CONTINUOUS, "season", "holiday", "workingday", "weather"]
    if data.empty or not set(required).issubset(data):
        raise ValueError("Empty input or missing required fields: " + ", ".join(required))
    data = data[required].copy()
    data["datetime"] = pd.to_datetime(data.datetime, errors="raise")
    t = data.datetime
    if t.dt.tz is not None or t.isna().any() or t.duplicated().any() or t.ne(t.dt.floor("h")).any():
        raise ValueError("Timestamps must be unique, naive, nonmissing whole hours")
    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError("Nonfinite values in " + col)
    if (data["count"].lt(0) | data["count"].mod(1).ne(0)).any():
        raise ValueError("Count must contain nonnegative integers")
    for field, values in dict(season=[1, 2, 3, 4], holiday=[0, 1],
                              workingday=[0, 1], weather=[1, 2, 3, 4]).items():
        if not data[field].isin(values).all():
            raise ValueError("Invalid category: " + field)
    if data[CONTINUOUS].lt(0).any().any() or data.humidity.gt(100).any():
        raise ValueError("Invalid weather variable range")
    data = data.sort_values("datetime").reset_index(drop=True)
    t = data.datetime
    if data.groupby(t.dt.normalize())[["season", "holiday", "workingday"]].nunique().gt(1).any().any():
        raise ValueError("Calendar metadata must be constant within dates")
    for col, values in dict(year=t.dt.year, month=t.dt.month, weekday=t.dt.dayofweek, hour=t.dt.hour).items():
        data[col] = values
    return data[NUMERIC + CATEGORICAL], data["count"], t


def partitions(t, test_fraction, n_splits):
    """Return row indices; inner fold indices are relative to outer training rows.

    Split observed dates, never individual hours. Equal numbers of observed dates
    need not span equal calendar durations because Kaggle omits days 20 onward.
    """
    if not 0 < test_fraction < 1 or n_splits < 2:
        raise ValueError("Require 0 < test fraction < 1 and at least two folds")
    dates = t.dt.normalize()
    unique = dates.drop_duplicates().to_numpy()
    n_train = len(unique) - int(np.ceil(len(unique)*test_fraction))
    if n_train < n_splits + 1 or len(unique) - n_train < 2:
        raise ValueError("Too few observed dates for the requested split/CV")
    train = np.flatnonzero(dates.isin(unique[:n_train]))
    test = np.flatnonzero(dates.isin(unique[n_train:]))
    train_dates = dates.iloc[train]
    folds = []
    for a, b in TimeSeriesSplit(n_splits=n_splits).split(unique[:n_train]):
        folds.append((np.flatnonzero(train_dates.isin(unique[a])),
                      np.flatnonzero(train_dates.isin(unique[b]))))
    return train, test, folds


def preprocessing():
    """Train-fitted numeric scaling and reference-level categorical encoding.

    Missing required inputs are rejected rather than silently imputed. Previously
    unseen categories get all zeros; they share the reference prediction and are
    explicitly counted in the coverage table. No ordinal weather interpretation.
    """
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=True)


def coefficient_table(model):
    prep = model.named_steps["preprocess"]
    sources = list(NUMERIC)
    encoder = prep.named_transformers_["category"]
    for col, categories in zip(CATEGORICAL, encoder.categories_):
        sources.extend([col] * (len(categories)-1))
    coefficients = model.named_steps["model"].coef_
    return pd.DataFrame(dict(encoded_feature=prep.get_feature_names_out(), feature=sources,
                             coefficient=coefficients, abs_coefficient=np.abs(coefficients),
                             selected=np.abs(coefficients) > TOLERANCE))


def forest(seed, trees):
    return Pipeline([("preprocess", preprocessing()), ("model", RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=1))])


def run(args):
    raw = pd.read_csv(args.input)
    x, y, t = prepare(raw)
    train, test, folds = partitions(t, args.test_fraction, args.cv_splits)
    xt, yt = x.iloc[train], y.iloc[train]
    xv, yv = x.iloc[test], y.iloc[test]
    print(f"Training {len(xt)} hours; final test {len(xv)} hours; tuning Lasso...", flush=True)
    # GridSearchCV encloses the WHOLE pipeline. Pipeline(preprocess, LassoCV)
    # would fit the scaler before LassoCV's internal folds and leak validation X.
    pipeline = Pipeline([("preprocess", preprocessing()),
                         ("model", Lasso(max_iter=100000, tol=1e-5))])
    search = GridSearchCV(pipeline, {"model__alpha": ALPHAS}, cv=folds,
                          scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(xt, yt)
    lasso = search.best_estimator_
    coef = coefficient_table(lasso)
    grouped = coef.groupby("feature").agg(lasso_abs_sum=("abs_coefficient", "sum"),
                                          lasso_nonzero=("selected", "sum"))
    grouped["lasso_rank"] = grouped.lasso_abs_sum.rank(method="min", ascending=False)

    # Primary importance is measured inside the training period. The final test
    # never determines any feature choice or alpha. RF hyperparameters are fixed.
    a, b = folds[-1]
    inspection_forest = forest(args.seed, args.trees).fit(xt.iloc[a], yt.iloc[a])
    print("Computing training-period validation permutation importance...", flush=True)
    importance = permutation_importance(inspection_forest, xt.iloc[b], yt.iloc[b],
                                        scoring="neg_root_mean_squared_error",
                                        n_repeats=args.repeats, random_state=args.seed, n_jobs=1)
    permutation = pd.DataFrame(dict(feature=x.columns, rmse_increase=importance.importances_mean,
                                    repeat_sd=importance.importances_std))
    permutation["permutation_rank"] = permutation.rmse_increase.rank(method="min", ascending=False)
    repeats = pd.DataFrame(importance.importances.T, columns=x.columns).rename_axis("repeat").reset_index()
    corr = pd.concat([xt[CONTINUOUS], yt], axis=1).corr(method="spearman")
    association = corr["count"].drop("count").rename("spearman_rho").to_frame()
    association["spearman_abs_rank"] = association.spearman_rho.abs().rank(method="min", ascending=False)
    comparison = pd.DataFrame(index=pd.Index(x.columns, name="feature")).join(association).join(grouped)
    comparison = comparison.join(permutation.set_index("feature"))
    comparison["lasso_selected"] = comparison.lasso_nonzero.gt(0)
    # A descriptive screening flag, not statistical significance. No reduced
    # model is fitted or claimed superior based on these flags.
    comparison["rf_positive_validation_importance"] = comparison.rmse_increase.gt(0)
    comparison = comparison.sort_values("permutation_rank").reset_index()

    final_forest = forest(args.seed, args.trees).fit(xt, yt)
    dummy = DummyRegressor(strategy="mean").fit(xt, yt)
    metrics = []
    predictions = pd.DataFrame(dict(datetime=t.iloc[test].to_numpy(), count=yv.to_numpy()))
    for name, model in [("training_mean", dummy), ("lasso", lasso), ("random_forest", final_forest)]:
        pred = model.predict(xv)
        predictions[name] = pred
        metrics.append(dict(model=name, split="final_test", n_hours=len(yv),
                            rmse=np.sqrt(mean_squared_error(yv, pred)),
                            mae=mean_absolute_error(yv, pred), r2=r2_score(yv, pred),
                            negative_predictions=int((pred < 0).sum())))
    val_pred = inspection_forest.predict(xt.iloc[b])
    metrics.append(dict(model="random_forest", split="inner_validation", n_hours=len(b),
                        rmse=np.sqrt(mean_squared_error(yt.iloc[b], val_pred)),
                        mae=mean_absolute_error(yt.iloc[b], val_pred), r2=r2_score(yt.iloc[b], val_pred),
                        negative_predictions=int((val_pred < 0).sum())))
    metrics = pd.DataFrame(metrics)
    cv = pd.DataFrame(search.cv_results_)
    cv_table = pd.DataFrame(dict(alpha=cv.param_model__alpha.astype(float),
                                mean_validation_mse=-cv.mean_test_score,
                                fold_mse_sd=cv.std_test_score, rank=cv.rank_test_score))
    for i in range(len(folds)):
        cv_table[f"fold_{i+1}_mse"] = -cv[f"split{i}_test_score"]
    boundaries, coverage = [], []
    for i, (a, b) in enumerate(folds, 1):
        for role, idx in [("train", a), ("validation", b)]:
            times = t.iloc[train[idx]]
            boundaries.append(dict(fold=i, role=role, start=str(times.min()), end=str(times.max()),
                                   hours=len(idx), observed_dates=times.dt.normalize().nunique()))
        for col in CATEGORICAL:
            coverage.append(dict(partition=f"cv_{i}", feature=col,
                                 unseen_hours=int((~xt.iloc[b][col].isin(xt.iloc[a][col])).sum())))
    for col in CATEGORICAL:
        coverage.append(dict(partition="final_test", feature=col,
                             unseen_hours=int((~xv[col].isin(xt[col])).sum())))
    splits = pd.DataFrame(dict(datetime=t, partition=np.where(np.arange(len(t)) < len(train), "train", "final_test")))
    encoder = lasso.named_steps["preprocess"].named_transformers_["category"]
    references = pd.DataFrame([dict(feature=col, reference=int(categories[0]),
                                    training_categories=";".join(map(str, categories)))
                               for col, categories in zip(CATEGORICAL, encoder.categories_)])
    tables = dict(spearman=corr.rename_axis("feature").reset_index(), lasso_coefficients=coef,
                  comparison=comparison, permutation=permutation, permutation_repeats=repeats,
                  model_metrics=metrics, cv_results=cv_table, cv_boundaries=pd.DataFrame(boundaries),
                  splits=splits, category_coverage=pd.DataFrame(coverage),
                  category_references=references, test_predictions=predictions)
    metadata = dict(input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    seed=args.seed, test_fraction=args.test_fraction, cv_splits=args.cv_splits,
                    train_hours=len(train), test_hours=len(test), train_dates=t.iloc[train].dt.normalize().nunique(),
                    test_dates=t.iloc[test].dt.normalize().nunique(),
                    train_start=str(t.iloc[train].min()), train_end=str(t.iloc[train].max()),
                    test_start=str(t.iloc[test].min()), test_end=str(t.iloc[test].max()),
                    numeric=NUMERIC, categorical=CATEGORICAL, ignored_input_columns=sorted(set(raw)-set(x)-{"datetime", "count"}),
                    alpha_grid=ALPHAS.tolist(), best_alpha=float(search.best_params_["model__alpha"]),
                    alpha_at_grid_boundary=bool(search.best_index_ in [0, len(ALPHAS)-1]),
                    alpha_scoring="mean fold MSE", coefficient_tolerance=TOLERANCE,
                    permutation_source="last inner validation block; forest fitted on preceding inner training block",
                    permutation_scoring="increase in RMSE (rentals/hour)", permutation_repeats=args.repeats,
                    rf_parameters=final_forest.named_steps["model"].get_params(), numerical_threads=1,
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                                  scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__))
    # Validate and compute everything before publishing outputs. Prior outputs
    # may still exist if a run fails; the metadata hash identifies the input.
    for path in [args.tables_dir, args.figures_dir, args.reports_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.tables_dir / f"08_{name}.csv", index=False, encoding="utf-8-sig")
    (args.tables_dir / "08_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    plots(corr, coef, comparison, cv_table, args.figures_dir)
    write_report(args, metadata, corr, comparison, metrics)
    print(metrics.to_string(index=False))
    print(comparison[["feature", "spearman_rho", "lasso_nonzero", "rmse_increase"]].to_string(index=False))


def plots(corr, coef, comparison, cv, directory):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Training-only Spearman association (exploratory)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(directory / "08_spearman.png", dpi=160)
    plt.close(fig)
    chosen = coef.loc[coef.selected].nlargest(25, "abs_coefficient").sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, max(4, len(chosen)*.26)))
    ax.barh(chosen.encoded_feature, chosen.coefficient)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(xlabel="Coefficient: numeric per training SD; dummy versus reference",
           title="Lasso: largest nonzero coefficients (up to 25)")
    fig.tight_layout()
    fig.savefig(directory / "08_lasso_coefficients.png", dpi=160)
    plt.close(fig)
    ordered = comparison.sort_values("rmse_increase")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered.feature, ordered.rmse_increase, xerr=ordered.repeat_sd, capsize=3)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(xlabel="Increase in validation RMSE (rentals/hour); bars = repeat SD",
           title="Random forest: training-period permutation importance")
    fig.tight_layout()
    fig.savefig(directory / "08_permutation_importance.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(cv.alpha, cv.mean_validation_mse, marker=".")
    best = cv.loc[cv["rank"].idxmin()]
    ax.axvline(best.alpha, color="red", linestyle="--", label=f"alpha={best.alpha:.4g}")
    ax.set(xlabel="Lasso alpha", ylabel="Mean chronological validation MSE", title="Lasso regularization selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "08_lasso_cv.png", dpi=160)
    plt.close(fig)


def write_report(args, meta, corr, comparison, metrics):
    """Generate only empirical statements computed during this run."""
    best = comparison.nlargest(3, "rmse_increase")
    associations = corr["count"].drop("count").sort_values(key=abs, ascending=False)
    lines = ["# Feature Selection（扩展三，第08步）", "",
             "本模块是原 EDA proposal 之外的预测分析扩展：比较统计关联、线性模型筛选和非线性模型依赖；不解释因果。", "",
             "## 数据与防泄漏", "",
             f"输入 SHA256：`{meta['input_sha256']}`。训练 {meta['train_hours']} 小时 / {meta['train_dates']} 日，"
             f"测试 {meta['test_hours']} 小时 / {meta['test_dates']} 日。",
             f"训练：{meta['train_start']} 至 {meta['train_end']}；测试：{meta['test_start']} 至 {meta['test_end']}。", "",
             "仅使用 temp、atemp、humidity、windspeed、year 及 one-hot 编码的 hour、weekday、month、season、holiday、workingday、weather。"
             "year 用数值趋势表示；时间字段从 datetime 重建。排除 casual、registered、用户占比、log1p_count、所有 flag、源行号等额外字段。"
             "保留通过校验的所有小时，不补缺失小时，不删除真实高需求或零天气值；字段缺失/非法时停止。", "",
             "## 方法", "",
             f"最后 {meta['test_fraction']:.0%} 的观测日期留作最终测试，同一天不跨分区。训练期用 {meta['cv_splits']} 折扩展窗口交叉验证。"
             "每折中重新拟合数值标准化和 one-hot 编码；GridSearchCV 搜索整个 Lasso pipeline，以平均验证 MSE 最低选择 alpha。"
             "类别参考值见 category_references 表；未见类别按全零编码，数量见 category_coverage。",
             f"本次 alpha={meta['best_alpha']:.6g}；是否位于搜索边界：{meta['alpha_at_grid_boundary']}。"
             "系数按原始 count 单位拟合，数值系数对应训练样本一个标准差；类别系数相对训练参考类别。"
             "abs_sum 为一个源变量所有编码系数绝对值之和，受类别数量、参考编码和共线性影响，不能与其他方法数值直接比较。", "",
             f"随机森林固定 {args.trees} 棵树、min_samples_leaf=2、种子 {args.seed}。重要性在最后一个训练期验证块计算，"
             f"森林仅拟合其之前的数据；每次打乱一个原始字段（类别的所有 dummy 一起改变），重复 {args.repeats} 次。"
             "正值表示 RMSE 增加。重复标准差描述置换随机性，不是日期抽样置信区间，也不是显著性检验。"
             "最终森林和 Lasso 在全部训练数据上拟合后各评估一次测试集。未用测试集排名、选择变量或调参。", "",
             "Spearman 只对训练期四个连续天气变量及 count 计算，无 p 值、无自动阈值筛选。分类编码不当连续量计算相关。", "",
             "## 实际结果", "", "训练期与 count 的 Spearman："]
    for feature, value in associations.items():
        lines.append(f"- {feature}：{value:.4f}。")
    lines += ["", f"temp 与 atemp 的 Spearman={corr.loc['temp', 'atemp']:.4f}；两者的替代信息会影响系数和置换排名。", "",
              f"Lasso 保留 {int(comparison.lasso_selected.sum())}/{len(comparison)} 个源变量的至少一个非零系数（阈值 {TOLERANCE:g}）。"
              "保留名单：" + "、".join(comparison.loc[comparison.lasso_selected, "feature"]) + "。", "",
              "训练期验证置换的重要性前三项："]
    for row in best.itertuples():
        lines.append(f"- {row.feature}：RMSE 增加 {row.rmse_increase:.3f} ± {row.repeat_sd:.3f}（重复标准差）。")
    lines += ["", "| 模型 | 分区 | RMSE | MAE | R² |", "| --- | --- | ---: | ---: | ---: |"]
    for row in metrics.itertuples():
        lines.append(f"| {row.model} | {row.split} | {row.rmse:.3f} | {row.mae:.3f} | {row.r2:.4f} |")
    lines += ["", "测试预测使用原始输出，不裁剪负预测；各模型负预测数见 model_metrics 表。训练均值模型提供基线。"
              "相关强度和预测价值可以不同；Lasso 为加性线性模型，森林可表达时间与工作日等交互。"
              "comparison 表支持比较，不合成一个所谓统一重要性分数；未训练缩减变量模型，因此不能声称筛选后精度提高。", "",
              "## 解读与局限", "",
              "建议把排名作为采集、解释和后续验证的依据，结合原 EDA 的小时/工作日曲线讨论调度。"
              "低置换值可能来自相关替代变量，不能证明该变量无价值。month/season 及 weekday/workingday/holiday 存在信息重叠；"
              "逐行置换也可能构造不自然的日历或天气组合。排名取决于模型与验证时期。", "",
              "原始 Kaggle 训练数据每月只覆盖 1–19 日；自定义输入的覆盖范围需另行核对。"
              "按观测日期分折的实际日历跨度可能不相等；缺失小时、时间依赖和季节分布变化均会影响解释。"
              "一次后期测试与单个验证块的重要性不足以代表全年或其他系统，不能把重复置换标准差当作泛化不确定性。"
              "类别极少（例如 weather=4）时无法推断一般模式。测试集未见类别采用参考预测是建模限制。"
              "此处使用同时段观测天气估计需求；提前预测需要相应预报天气，误差未在此验证。"
              "本模块未读取无 count 的 Kaggle test.csv，不生成或伪造其标签。", "",
              "## 复现与产物", "", "```powershell", "python -m pip install -r requirements.txt",
              "python scripts/08_feature_selection.py", "python -m unittest discover -s scripts -p 'test_08_feature_selection.py'", "```", "",
              "默认输出：outputs/tables/08_*（12 个 CSV 和运行元数据 JSON），outputs/figures/08_*（4 张 PNG），"
              "reports/feature_selection.md（本自动报告；重跑覆盖）。用 --help 查看输入/表图/报告目录、种子、测试比例、折数、树数与重复次数。"
              "变更参数属于新实验，勿按测试表现反复调整。校验失败时旧产物可能仍存在，需核对运行状态和输入哈希。", "",
              "本次环境：" + "；".join(f"{key} {value}" for key, value in meta["versions"].items()) + "。", "",
              "实现方案见 docs/feature_selection_plan.md。方法参考："
              "[scikit-learn permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html)、"
              "[TimeSeriesSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。"]
    (args.reports_dir / "feature_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train_eda.csv")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.trees < 1 or args.repeats < 2:
        parser.error("Trees must be positive and permutation repeats >=2")
    if (args.input.name.startswith("08_") and args.input.resolve().parent in
            [args.tables_dir.resolve(), args.figures_dir.resolve()]) or args.input.resolve() == (args.reports_dir / "feature_selection.md").resolve():
        parser.error("Input must be separate from generated outputs")
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            # Coverage is exported per fold/field instead of repeating this
            # expected warning for every alpha and every permutation.
            warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
            run(args)
    except (OSError, ValueError, ConvergenceWarning) as exc:
        print(f"Feature selection stopped: {exc}. Existing outputs may be from an older run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
