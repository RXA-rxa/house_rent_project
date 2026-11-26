# 基于 PySpark 的北京租房价格预测分析

使用 PySpark（Spark MLlib）对北京租房挂牌数据进行清洗、特征工程与回归建模，对比线性回归与随机森林两种模型的预测效果，并通过可视化呈现租金分布、区县差异、面积与租金关系及特征重要性。

## 项目结构

```
house_rent_project/
├── data/                          # 原始租房数据（bj_danke_1 ~ 8.csv）
├── outputs/                       # 运行后生成的分析图表
├── house_rent_analysis.py         # 主分析脚本（命令行 / 脚本运行）
├── house_rent_analysis.ipynb      # 同一套代码的 Jupyter 笔记本版本
└── 北京租房价格分析报告.docx       # 课程设计报告
```

## 数据说明

数据为北京各区域二手房/租房挂牌信息，主要字段包括：价格、面积、位置1（区县）、户型、楼层、小区、地铁等。脚本从「户型」「楼层」「地铁」等文本字段中正则提取室数、卫数、当前楼层、总楼层、地铁距离等数值特征。

## 分析流程

1. **数据读取**：通配符读取 `data/bj_danke_*.csv` 全部文件并合并。
2. **数据清洗**：去重、统计缺失值、正则拆分数值特征、过滤价格与面积异常范围。
3. **缺失值填充**：数值列用训练集中位数填充，区县列缺失填「未知」。
4. **特征工程**：区县做 StringIndexer + OneHotEncoder 编码，数值列与编码列经 VectorAssembler 组装后用 MinMaxScaler 归一化，封装进 Pipeline。
5. **模型训练**：按 8:2 划分训练集与测试集，分别训练线性回归与随机森林回归。
6. **模型评估**：使用 RMSE、MAE、R² 三项指标对比两个模型。
7. **可视化**：输出租金分布、各区县平均租金、面积-租金散点、预测值 vs 真实值、特征重要性共 5 张图。

## 环境要求

- Python 3.10
- Java 8（与 Spark 3.3.x 配套）
- 依赖库：`pyspark==3.3.4`、`matplotlib`、`numpy`、`pandas`、`ipykernel`

安装依赖：

```bash
pip install pyspark==3.3.4 matplotlib numpy pandas ipykernel
```

## 运行方式

命令行运行脚本：

```bash
python house_rent_analysis.py
```

或在 JupyterLab / Jupyter Notebook 中打开 `house_rent_analysis.ipynb` 依次运行单元格。运行结束后图表保存到 `outputs/` 目录。

## 主要结果（本机测试集）

| 模型 | RMSE（元） | MAE（元） | R² |
| --- | --- | --- | --- |
| 线性回归 | 470.29 | 327.59 | 0.8352 |
| 随机森林 | 440.09 | 319.77 | 0.8557 |

- 面积与租金相关系数：0.8075
- 平均租金约 2662 元，中位数 2430 元
- 测试集上随机森林效果优于线性回归

## 特征说明

数值特征：面积、室数、卫数、当前楼层、总楼层、地铁距离；分类特征：位置1（区县，独热编码）。
