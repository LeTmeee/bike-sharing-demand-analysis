# 共享单车需求分析项目

**CityU 5002｜Group 18（C62）**

使用 Bike Sharing Demand 数据，分析需求的时间规律、天气关联、用户构成和数据质量，并通过假设检验、聚类和特征选择补充解释。主数据包含2011—2012年10,886条小时记录；实现采用Python脚本。

阅读结论请先打开 **[最终报告](reports/final_report.md)**；查看自动汇总请打开 **[综合摘要](reports/synthesis_summary.md)**。

## 1. 研究问题

| 内容 | 要回答的问题                     |
| --- |----------------------------|
| RQ1：时间规律 | 需求如何随小时、星期、工作日状态、月份和季节变化？  |
| RQ2：天气关联 | 不同天气类别及温度、湿度、风速与需求有什么关联？   |
| RQ3：用户行为 | 注册和临时租赁的数量、占比及日内模式有何不同？    |
| RQ4：数据质量与分布 | 数据是否存在缺失、重复、非法值、时间缺口或极端需求？ |
| 扩展一：假设检验 | 日层面的工作日、天气差异和温度关联是否有统计证据？  |
| 扩展二：聚类 | 每日24小时需求曲线能否分为不同的形状和规模模式？  |
| 扩展三：特征选择 | 哪些日历、天气变量对估计小时需求更有帮助？      |

## 2. 项目流程与脚本功能

```text
原始数据 → 01 数据审计 → 02 校验与统一处理 → train_eda.csv
                                           ├→ 03 时间规律
                                           ├→ 04 天气关联
                                           ├→ 05 用户行为
                                           ├→ 06 假设检验
                                           ├→ 07 聚类分析
                                           └→ 08 特征选择
原始文件 + 当前处理数据 + 06–08结果 → 09 综合分析 → synthesis_summary.md
各模块证据 + 图表解读 + 综合讨论 → final_report.md
```

| 步骤 | scripts中的脚本 | 功能与主要产物 |
| --- | --- | --- |
| 01 | `01_data_audit.py` | 检查三个原始CSV的结构、缺失、重复、取值、计数恒等式和时间覆盖，输出`audit_*`表 |
| 02 | `02_distribution_quality.py` | 校验训练数据、绘制分布、标记异常候选、派生时间等字段，生成`train_eda.csv`及`02_*`表图 |
| 03 | `03_temporal_patterns.py` | 小时、星期、工作日、周末、月份与季节汇总，输出`03_*`表图 |
| 04 | `04_weather_analysis.py` | 天气类别比较、散点、分箱、相关、分层及零值敏感性，输出`04_*`表图 |
| 05 | `05_user_behavior.py` | 比较注册与临时租赁的数量、占比和时间模式，输出`05_*`表图 |
| 06 | `06_hypothesis_testing.py` | 日期聚合、非参数检验、效应量、Holm校正和bootstrap区间，输出`06_*`表图及运行元数据 |
| 07 | `07_clustering.py` | 完整日期的24小时KMeans聚类，比较原始数量与日内占比，输出`07_*`表图及运行元数据 |
| 08 | `08_feature_selection.py` | 训练期Spearman、时间交叉验证Lasso、森林置换重要性与后期测试，输出`08_*`表图、元数据及特征报告 |
| 09 | `09_synthesis.py` | 重算基础EDA核心证据、重新审计原始文件，核对06–08输入哈希，输出`09_*`表图、元数据及综合摘要 |

先完成01、02，再运行03–08，最后运行09。03–08独立读取处理后数据；09需要06–08已生成且与当前输入版本一致的结果。09还会导入01和05的辅助函数，因此请保持整个`scripts/`目录完整。

## 3. 环境配置（Windows PowerShell / macOS zsh、bash）

环境使用 **Python 3.11**。所有命令均在项目根目录执行。先确认当前Python版本，再创建独立环境；统一使用`.venv`作为虚拟环境目录。

**Windows PowerShell：**

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn, threadpoolctl; print('Dependencies OK')"
```

**macOS（zsh / bash）：**

先在终端进入项目根目录（包含 `README.md`、`requirements.txt` 和 `scripts/` 的目录），后续所有命令均从该目录执行。检查是否已安装 Python 3.11：

```bash
python3.11 --version
```

应显示 `Python 3.11.x`。如果提示找不到命令，请先安装 Python 3.11：可使用 Python 官网的 macOS 3.11 安装包；如果已安装 Homebrew，也可运行以下命令。安装完成后重新打开终端，回到项目根目录并再次检查版本。

```bash
# 仅在已安装 Homebrew 且需要安装 Python 3.11 时执行
brew install python@3.11
python3.11 --version
```

确认版本后创建虚拟环境、升级 pip、安装并验证依赖：

```bash
python3.11 -m venv .venv
./.venv/bin/python --version
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn, threadpoolctl; print('Dependencies OK')"
```

`pip check` 应显示依赖兼容检查通过，导入检查应输出 `Dependencies OK`。macOS 后续命令统一使用 `./.venv/bin/python`，无需激活虚拟环境。

依赖以[requirements.txt](requirements.txt)为准：pandas、NumPy用于数据处理，Matplotlib、Seaborn用于绘图，SciPy用于统计检验，scikit-learn用于聚类和建模，threadpoolctl用于控制数值线程。当前没有完整依赖版本锁定，06–09元数据记录对应运行版本。

下方命令直接调用虚拟环境解释器，**无需激活环境**。若使用其他已装齐依赖的环境，统一替换解释器路径即可；不要在同一次流程中混用环境。

## 4. 运行命令

### 4.1 按顺序运行分析

以下 Windows 和 macOS 命令均从项目根目录执行。

**Windows PowerShell：**

```powershell
.\.venv\Scripts\python.exe scripts/01_data_audit.py
.\.venv\Scripts\python.exe scripts/02_distribution_quality.py
.\.venv\Scripts\python.exe scripts/03_temporal_patterns.py
.\.venv\Scripts\python.exe scripts/04_weather_analysis.py
.\.venv\Scripts\python.exe scripts/05_user_behavior.py
.\.venv\Scripts\python.exe scripts/06_hypothesis_testing.py
.\.venv\Scripts\python.exe scripts/07_clustering.py
.\.venv\Scripts\python.exe scripts/08_feature_selection.py
.\.venv\Scripts\python.exe scripts/09_synthesis.py
```

**macOS（zsh / bash）：**

```bash
./.venv/bin/python scripts/01_data_audit.py
./.venv/bin/python scripts/02_distribution_quality.py
./.venv/bin/python scripts/03_temporal_patterns.py
./.venv/bin/python scripts/04_weather_analysis.py
./.venv/bin/python scripts/05_user_behavior.py
./.venv/bin/python scripts/06_hypothesis_testing.py
./.venv/bin/python scripts/07_clustering.py
./.venv/bin/python scripts/08_feature_selection.py
./.venv/bin/python scripts/09_synthesis.py
```

**请逐条运行。** 每步结束后，Windows PowerShell 可立即输入`$LASTEXITCODE`，macOS zsh / bash 可立即输入`echo $?`检查状态，0表示正常退出；非0时先处理错误再继续，避免后续步骤使用旧文件。审计中的WARN需要复核，不一定表示数据错误。

如果数据和06–08结果均未改变，只需重新生成综合摘要：

**Windows PowerShell：**

```powershell
.\.venv\Scripts\python.exe scripts/09_synthesis.py
```

**macOS（zsh / bash）：**

```bash
./.venv/bin/python scripts/09_synthesis.py
```

### 4.2 测试与参数

**Windows PowerShell：**

```powershell
# 运行06–09模块的全部自动化测试
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_*.py"

# 查看各脚本支持的参数
.\.venv\Scripts\python.exe scripts/08_feature_selection.py --help
.\.venv\Scripts\python.exe scripts/09_synthesis.py --help
```

**macOS（zsh / bash）：**

```bash
# 运行06–09模块的全部自动化测试
./.venv/bin/python -m unittest discover -s scripts -p "test_*.py"

# 查看各脚本支持的参数
./.venv/bin/python scripts/08_feature_selection.py --help
./.venv/bin/python scripts/09_synthesis.py --help
```

脚本默认从项目位置定位输入和输出。各脚本路径参数略有不同，使用前查看`--help`：01使用`--data-dir`、`--output-dir`；02支持`--processed-dir`；03–08通常支持`--input`、`--tables-dir`、`--figures-dir`。09另外支持`--raw-dir`和`--source-tables-dir`，分别指定原始文件及06–08证据目录。

重跑会覆盖同名生成产物。08会更新`feature_selection.md`，09会更新`synthesis_summary.md`；`final_report.md`、`hypothesis_testing.md`和`clustering.md`需在结果变化后人工同步。更改输入后必须重跑受影响的分析；09发现06–08输入哈希不一致时会停止。

## 5. 项目目录与文件说明

```text
project/
├── .gitignore                          # 排除本地环境、缓存和临时文件
├── .gitattributes                      # 文本换行规则与CSV字节保护
├── README.md                           # 项目说明与运行入口
├── requirements.txt                    # Python依赖清单
├── data/
│   ├── README.md                       # 数据来源、标签与分享说明
│   ├── raw/
│   │   ├── train.csv                   # 有真实需求标签的原始训练数据
│   │   ├── test.csv                    # 无需求标签的竞赛测试数据
│   │   └── sampleSubmission.csv        # 提交格式示例，不是真实需求
│   └── processed/
│       └── train_eda.csv               # 02生成的统一分析输入
├── scripts/
│   ├── 01_data_audit.py
│   ├── 02_distribution_quality.py
│   ├── 03_temporal_patterns.py
│   ├── 04_weather_analysis.py
│   ├── 05_user_behavior.py
│   ├── 06_hypothesis_testing.py
│   ├── 07_clustering.py
│   ├── 08_feature_selection.py
│   ├── 09_synthesis.py
│   ├── test_06_hypothesis_testing.py    # 日期聚合、统计辅助函数等测试
│   ├── test_07_clustering.py            # 聚类输入、覆盖、输出一致性等测试
│   ├── test_08_feature_selection.py     # 防泄漏、时间划分、预处理等测试
│   └── test_09_synthesis.py             # 上游证据、输入哈希、报告链接等测试
├── outputs/
│   ├── tables/                         # audit_*、02_*至09_*的CSV与运行JSON
│   └── figures/                        # 02_*至09_*的PNG图表
├── reports/
│   ├── final_report.md                 # 整体最终报告、图表解读和简短结论
│   ├── synthesis_summary.md            # 09自动生成的前八步证据摘要
│   ├── hypothesis_testing.md           # 检验设计、效应量与结果说明
│   ├── clustering.md                   # 聚类设计、曲线模式与结果说明
│   └── feature_selection.md            # 08自动生成的特征与模型结果报告
└── docs/
    ├── hypothesis_testing_plan.md      # 假设检验实现方案
    ├── clustering_plan.md              # 聚类实现方案
    ├── feature_selection_plan.md       # 特征选择实现方案
    └── synthesis_reorganization_plan.md # 模块改号与最终综合方案
```

目录树省略了本地环境和临时文件：`.venv/`为解释器与依赖，`.idea/`为IDE配置，`__pycache__/`为Python缓存，`tmp/`为历史临时或检查产物。它们不作为分析结论的证据入口。

结果文件通常以步骤编号作为前缀，例如`08_model_metrics.csv`对应第08步模型指标，`07_share_profiles.png`对应第07步日内占比聚类曲线。`*_run_metadata.json`记录输入哈希、设置和运行信息；CSV保存详细证据，PNG用于展示，Markdown报告负责解释。

## 6. 阅读与协作注意事项

- **看整体结论：**阅读[final_report.md](reports/final_report.md)，各章、图和表都有简短结论；附录索引覆盖全部38张图。
- **查专题方法：**阅读[假设检验](reports/hypothesis_testing.md)、[聚类](reports/clustering.md)和[特征选择](reports/feature_selection.md)报告；具体数字回查`outputs/tables/`。
- **保持统一数据：**不手工改动`data/raw/`、处理数据或生成表。需改变规则时修改对应代码，再重跑下游分析。
- **区分标签与缺口：**训练标签仅覆盖每月1–19日；缺失小时不补零，提交模板中的零值不参与分析。
- **避免错误解释：**相关不代表因果，租赁次数不等于人数；预测count时不得使用casual、registered或需求衍生字段。
- **核对生成状态：**运行失败后旧文件可能仍在；结合退出码、输入哈希和运行元数据判断是否为本次结果。
