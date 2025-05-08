# PySpark Distributed Data Processing Example

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Spark session
spark = SparkSession.builder.appName("DistributedModelTraining").getOrCreate()

# Load large dataset
df = spark.read.csv("large_data.csv", header=True, inferSchema=True)

# Data transformation
df = df.dropna()
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
df = assembler.transform(df)
df = df.withColumnRenamed(df.columns[-1], "label")

# Split data
train, test = df.randomSplit([0.8, 0.2])

# Model training
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
model = rf.fit(train)

# Evaluate
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
print("AUC:", evaluator.evaluate(predictions))

spark.stop()
