# Feature Selection（扩展三，第08步）

本模块是原 EDA proposal 之外的预测分析扩展：比较统计关联、线性模型筛选和非线性模型依赖；不解释因果。

## 数据与防泄漏

输入 SHA256：`e5305c1d3624a7c0401ac35bbe402131a00ae8e68429afdc78a140f9cc171c51`。训练 8679 小时 / 364 日，测试 2207 小时 / 92 日。
训练：2011-01-01 00:00:00 至 2012-08-03 23:00:00；测试：2012-08-04 00:00:00 至 2012-12-19 23:00:00。

仅使用 temp、atemp、humidity、windspeed、year 及 one-hot 编码的 hour、weekday、month、season、holiday、workingday、weather。year 用数值趋势表示；时间字段从 datetime 重建。排除 casual、registered、用户占比、log1p_count、所有 flag、源行号等额外字段。保留通过校验的所有小时，不补缺失小时，不删除真实高需求或零天气值；字段缺失/非法时停止。

## 方法

最后 20% 的观测日期留作最终测试，同一天不跨分区。训练期用 5 折扩展窗口交叉验证。每折中重新拟合数值标准化和 one-hot 编码；GridSearchCV 搜索整个 Lasso pipeline，以平均验证 MSE 最低选择 alpha。类别参考值见 category_references 表；未见类别按全零编码，数量见 category_coverage。
本次 alpha=0.14678；是否位于搜索边界：False。系数按原始 count 单位拟合，数值系数对应训练样本一个标准差；类别系数相对训练参考类别。abs_sum 为一个源变量所有编码系数绝对值之和，受类别数量、参考编码和共线性影响，不能与其他方法数值直接比较。

随机森林固定 150 棵树、min_samples_leaf=2、种子 18。重要性在最后一个训练期验证块计算，森林仅拟合其之前的数据；每次打乱一个原始字段（类别的所有 dummy 一起改变），重复 10 次。正值表示 RMSE 增加。重复标准差描述置换随机性，不是日期抽样置信区间，也不是显著性检验。最终森林和 Lasso 在全部训练数据上拟合后各评估一次测试集。未用测试集排名、选择变量或调参。

Spearman 只对训练期四个连续天气变量及 count 计算，无 p 值、无自动阈值筛选。分类编码不当连续量计算相关。

## 实际结果

训练期与 count 的 Spearman：
- atemp：0.4412。
- temp：0.4393。
- humidity：-0.3569。
- windspeed：0.1345。

temp 与 atemp 的 Spearman=0.9948；两者的替代信息会影响系数和置换排名。

Lasso 保留 10/12 个源变量的至少一个非零系数（阈值 1e-08）。保留名单：hour、humidity、atemp、weekday、temp、weather、month、season、windspeed、year。

训练期验证置换的重要性前三项：
- hour：RMSE 增加 146.746 ± 2.945（重复标准差）。
- workingday：RMSE 增加 39.517 ± 1.710（重复标准差）。
- humidity：RMSE 增加 38.687 ± 2.136（重复标准差）。

| 模型 | 分区 | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| training_mean | final_test | 233.589 | 176.498 | -0.1567 |
| lasso | final_test | 133.415 | 99.701 | 0.6227 |
| random_forest | final_test | 107.167 | 73.771 | 0.7565 |
| random_forest | inner_validation | 96.729 | 70.496 | 0.8051 |

测试预测使用原始输出，不裁剪负预测；各模型负预测数见 model_metrics 表。训练均值模型提供基线。相关强度和预测价值可以不同；Lasso 为加性线性模型，森林可表达时间与工作日等交互。comparison 表支持比较，不合成一个所谓统一重要性分数；未训练缩减变量模型，因此不能声称筛选后精度提高。

## 解读与局限

建议把排名作为采集、解释和后续验证的依据，结合原 EDA 的小时/工作日曲线讨论调度。低置换值可能来自相关替代变量，不能证明该变量无价值。month/season 及 weekday/workingday/holiday 存在信息重叠；逐行置换也可能构造不自然的日历或天气组合。排名取决于模型与验证时期。

原始 Kaggle 训练数据每月只覆盖 1–19 日；自定义输入的覆盖范围需另行核对。按观测日期分折的实际日历跨度可能不相等；缺失小时、时间依赖和季节分布变化均会影响解释。一次后期测试与单个验证块的重要性不足以代表全年或其他系统，不能把重复置换标准差当作泛化不确定性。类别极少（例如 weather=4）时无法推断一般模式。测试集未见类别采用参考预测是建模限制。此处使用同时段观测天气估计需求；提前预测需要相应预报天气，误差未在此验证。本模块未读取无 count 的 Kaggle test.csv，不生成或伪造其标签。

## 复现与产物

```powershell
python -m pip install -r requirements.txt
python scripts/08_feature_selection.py
python -m unittest discover -s scripts -p 'test_08_feature_selection.py'
```

默认输出：outputs/tables/08_*（12 个 CSV 和运行元数据 JSON），outputs/figures/08_*（4 张 PNG），reports/feature_selection.md（本自动报告；重跑覆盖）。用 --help 查看输入/表图/报告目录、种子、测试比例、折数、树数与重复次数。变更参数属于新实验，勿按测试表现反复调整。校验失败时旧产物可能仍存在，需核对运行状态和输入哈希。

本次环境：python 3.8.17；pandas 2.0.3；numpy 1.24.4；scipy 1.10.1；sklearn 1.3.2；matplotlib 3.7.5。

实现方案见 docs/feature_selection_plan.md。方法参考：[scikit-learn permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html)、[TimeSeriesSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。
