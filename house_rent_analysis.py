# -*- coding: utf-8 -*-
"""北京租房价格预测"""

import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    MinMaxScaler,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
   BASE_DIR = Path(r"D:\house_rent_project")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

spark = (
    SparkSession.builder.master("local[4]")
    .appName("HouseRentAnalysis")
    .config("spark.driver.memory", "2g")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


# 读取八个数据文件
files = sorted(glob.glob(str(BASE_DIR / "data" / "bj_danke_*.csv")))
df = spark.read.csv(files, header=True, inferSchema=True, encoding="UTF-8")
print(f"原始数据：{df.count()} 行，{len(df.columns)} 列")


# 去重、修复类型并过滤异常范围
df = df.dropDuplicates()
print(f"去重后：{df.count()} 行")
missing = df.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
).first().asDict()
print("缺失值：", {key: value for key, value in missing.items() if value})

# 文本字段拆成数值特征
df = df.withColumn(
    "室数", F.regexp_extract("户型", r"(\d+)室", 1).cast("int")
)
df = df.withColumn(
    "卫数", F.regexp_extract("户型", r"(\d+)卫", 1).cast("int")
)
df = df.withColumn(
    "当前楼层", F.regexp_extract("楼层", r"(\d+)/", 1).cast("int")
)
df = df.withColumn(
    "总楼层", F.regexp_extract("楼层", r"/(\d+)层", 1).cast("int")
)
df = df.withColumn(
    "地铁距离", F.regexp_extract("地铁", r"(\d+)米", 1).cast("int")
)

df = df.filter(
    F.col("价格").between(500, 30000)
    & F.col("面积").between(5, 500)
)
print(f"清洗后：{df.count()} 行")
df.select("价格", "面积").summary(
    "count", "min", "25%", "50%", "75%", "max"
).show()


label_col = "价格"
num_cols = ["面积", "室数", "卫数", "当前楼层", "总楼层", "地铁距离"]
model_data = df.select(label_col, *num_cols, "位置1")
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
print(f"训练集 {train_data.count()} 行，测试集 {test_data.count()} 行")

# 缺失值用训练集中位数填充
medians = {
    column: train_data.approxQuantile(column, [0.5], 0.001)[0]
    for column in num_cols
}
train_data = train_data.na.fill(medians).na.fill({"位置1": "未知"})
test_data = test_data.na.fill(medians).na.fill({"位置1": "未知"})

pipeline = Pipeline(
    stages=[
        StringIndexer(
            inputCol="位置1",
            outputCol="位置1_idx",
            handleInvalid="keep",
        ),
        OneHotEncoder(
            inputCol="位置1_idx",
            outputCol="位置1_onehot",
            handleInvalid="keep",
        ),
        VectorAssembler(
            inputCols=num_cols + ["位置1_onehot"],
            outputCol="features_raw",
        ),
        MinMaxScaler(inputCol="features_raw", outputCol="features"),
    ]
)
prep_model = pipeline.fit(train_data)
train_prepared = prep_model.transform(train_data)
test_prepared = prep_model.transform(test_data)


# 训练监督学习模型
lr_model = LinearRegression(
    featuresCol="features",
    labelCol=label_col,
    maxIter=100,
    regParam=0.01,
).fit(train_prepared)
lr_test = lr_model.transform(test_prepared)

rf_model = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col,
    numTrees=50,
    maxDepth=8,
    seed=42,
).fit(train_prepared)
rf_test = rf_model.transform(test_prepared)

evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse",
)


def scores(prediction):
    rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(prediction, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(prediction, {evaluator.metricName: "r2"})
    return rmse, mae, r2


lr_rmse, lr_mae, lr_r2 = scores(lr_test)
rf_rmse, rf_mae, rf_r2 = scores(rf_test)
print("\n模型测试结果")
print(f"线性回归  RMSE={lr_rmse:.2f}  MAE={lr_mae:.2f}  R2={lr_r2:.4f}")
print(f"随机森林  RMSE={rf_rmse:.2f}  MAE={rf_mae:.2f}  R2={rf_r2:.4f}")

best_test = rf_test if rf_rmse < lr_rmse else lr_test
best_name = "随机森林" if best_test is rf_test else "线性回归"



price_pd = df.select("价格").toPandas()
mean_price = df.select(F.mean("价格")).first()[0]
median_price = df.approxQuantile("价格", [0.5], 0.001)[0]

plt.figure(figsize=(9, 5.5))
plt.hist(price_pd["价格"], bins=40, color="#4C78A8", edgecolor="white")
plt.axvline(mean_price, color="#D1495B", linestyle="--", label="均值")
plt.axvline(median_price, color="#2A9D8F", label="中位数")
plt.title("北京租房月租金分布")
plt.xlabel("月租金（元）")
plt.ylabel("房源数量")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_price_distribution.png")
plt.close()

district_pd = (
    df.groupBy("位置1")
    .agg(F.mean("价格").alias("平均租金"))
    .orderBy("平均租金")
    .toPandas()
)
plt.figure(figsize=(9, 6))
plt.barh(district_pd["位置1"], district_pd["平均租金"], color="#4C78A8")
plt.title("北京各区县平均月租金")
plt.xlabel("平均月租金（元）")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_district_rent.png")
plt.close()

area_price_pd = df.select("面积", "价格").toPandas()
correlation = df.stat.corr("面积", "价格")
coefficients = np.polyfit(area_price_pd["面积"], area_price_pd["价格"], 1)
trend = np.poly1d(coefficients)
line_x = np.linspace(area_price_pd["面积"].min(), area_price_pd["面积"].max(), 100)

plt.figure(figsize=(9, 6))
plt.scatter(
    area_price_pd["面积"],
    area_price_pd["价格"],
    s=14,
    alpha=0.35,
    color="#4C78A8",
)
plt.plot(line_x, trend(line_x), color="#D1495B", linestyle="--")
plt.title(f"面积与月租金关系（相关系数 {correlation:.2f}）")
plt.xlabel("面积（平方米）")
plt.ylabel("月租金（元）")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_area_vs_price.png")
plt.close()

prediction_pd = best_test.select(
    F.col(label_col).alias("真实值"),
    F.col("prediction").alias("预测值"),
).toPandas()
limit = max(prediction_pd.max()) * 1.05

plt.figure(figsize=(7.5, 7))
plt.scatter(
    prediction_pd["真实值"],
    prediction_pd["预测值"],
    s=16,
    alpha=0.4,
    color="#4C78A8",
)
plt.plot([0, limit], [0, limit], color="#D1495B", linestyle="--")
plt.xlim(0, limit)
plt.ylim(0, limit)
plt.title(f"{best_name}测试集预测结果")
plt.xlabel("真实月租金（元）")
plt.ylabel("预测月租金（元）")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_prediction_vs_actual.png")
plt.close()

importance = list(rf_model.featureImportances)
feature_names = num_cols + ["区县"]
feature_values = importance[: len(num_cols)] + [sum(importance[len(num_cols) :])]
order = np.argsort(feature_values)
plot_names = [feature_names[index] for index in order]
plot_values = [feature_values[index] for index in order]

plt.figure(figsize=(8, 5))
plt.barh(plot_names, plot_values, color="#4C78A8")
plt.title("随机森林特征重要性")
plt.xlabel("重要性得分")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_feature_importance.png")
plt.close()


print(f"\n平均租金 {mean_price:.0f} 元，中位数 {median_price:.0f} 元")
print(f"面积与租金相关系数：{correlation:.4f}")
print(f"测试集效果较好的模型：{best_name}")
print(f"图片保存到：{OUTPUT_DIR}")

spark.stop()
