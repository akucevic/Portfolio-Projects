from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ETL').getOrCreate()
df = spark.read.csv('data.csv', header=True, inferSchema=True)

df_clean = df.dropna()
agg = df_clean.groupBy('category').count()
agg.show()
