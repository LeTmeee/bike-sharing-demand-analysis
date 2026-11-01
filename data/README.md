# 数据来源与文件说明

项目使用 Kaggle **Bike Sharing Demand** 竞赛数据，来源入口：
[Bike Sharing Demand / Data](https://www.kaggle.com/competitions/bike-sharing-demand/data)。

| 文件 | 本地结构 | 用途 |
| --- | --- | --- |
| raw/train.csv | 10,886行、12列 | 真实小时租赁量与日历、天气、用户类别 |
| raw/test.csv | 6,493行、9列 | 无需求标签，只用于字段及时间覆盖审计 |
| raw/sampleSubmission.csv | 6,493行、2列 | 提交格式；count中的零不是实际需求 |
| processed/train_eda.csv | 10,886行、34列 | 第02步生成，保留原始记录并增加派生字段和复核标记 |

当前目录没有保存数据下载时的条款或独立数据许可文件，因此这里不声称拥有数据的再分发授权。
公开上传原始和处理后数据前，请核对来源的分享条件。若选择不随仓库提供数据，
请让组员从来源获取三个原始CSV并放入raw/，依次运行01—09重新生成结果。
当前09集成测试还会读取共享结果表，因此不能只删除所有表而不调整测试准备流程。

字段名称与计数定义以项目代码的实际校验为准。season/weather暂保留数字编码，
温度等变量保留源数值；展示前若增加类别名称和单位，应先核对来源说明。
缺失小时不补零；不把casual、registered及需求衍生字段用于预测count。
