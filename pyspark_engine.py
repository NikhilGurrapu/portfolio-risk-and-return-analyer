from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, mean, stddev, lit
from pyspark.sql.window import Window
import yfinance as yf
import pandas as pd
import numpy as np

# Initialize Spark session
def initialize_spark():
    return SparkSession.builder \
        .appName("Portfolio Analyzer") \
        .getOrCreate()

# Fetch price data using yfinance and convert to Spark DataFrame
def fetch_prices_spark(spark, tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    df = df.fillna(method='ffill')
    df = df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Price')
    sdf = spark.createDataFrame(df)
    return sdf

# Calculate performance metrics and return cumulative returns
def calculate_metrics_spark(price_df, weights):
    window_spec = Window.partitionBy("Ticker").orderBy("Date")

    price_df = price_df.withColumn("Prev_Price", lag("Price").over(window_spec))
    price_df = price_df.withColumn("Daily_Return", (col("Price") - col("Prev_Price")) / col("Prev_Price"))
    price_df = price_df.dropna()

    tickers = price_df.select("Ticker").distinct().rdd.flatMap(lambda x: x).collect()
    weight_dict = dict(zip(tickers, weights))
    w_df = price_df.withColumn("Weight", col("Ticker").apply(lambda x: float(weight_dict.get(x, 0))))
    
    weighted = w_df.withColumn("Weighted_Return", col("Daily_Return") * col("Weight"))
    port_df = weighted.groupBy("Date").sum("Weighted_Return").withColumnRenamed("sum(Weighted_Return)", "Portfolio_Return")

    port_df = port_df.withColumn("Cumulative_Return", 
                                 (lit(1) + col("Portfolio_Return")).cast("double"))
    
    # Calculate cumulative return
    pdf = port_df.toPandas()
    pdf["Cumulative_Return"] = (1 + pdf["Portfolio_Return"]).cumprod()
    
    # Annualized metrics
    avg_daily_return = pdf["Portfolio_Return"].mean()
    daily_volatility = pdf["Portfolio_Return"].std()
    sharpe_ratio = (avg_daily_return / daily_volatility) * np.sqrt(252)
    annual_return = (1 + avg_daily_return) ** 252 - 1
    annual_volatility = daily_volatility * np.sqrt(252)

    result_df = pd.DataFrame({
        "Date": pdf["Date"],
        "Cumulative_Return": pdf["Cumulative_Return"]
    })

    return annual_return, annual_volatility, sharpe_ratio, spark.createDataFrame(result_df)
