from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("TestSession").getOrCreate()

print("✅ PySpark is working!")
print("Spark Version:", spark.version)

# Stop Spark Session
spark.stop()
