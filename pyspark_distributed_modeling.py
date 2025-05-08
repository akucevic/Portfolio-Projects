from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('DistML').getOrCreate()
df = spark.read.csv('large_data.csv', header=True, inferSchema=True)

assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol='features')
df_vec = assembler.transform(df)
rf = RandomForestClassifier(labelCol='label', featuresCol='features')
model = rf.fit(df_vec)
