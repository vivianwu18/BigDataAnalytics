import findspark
from pyspark.sql import SparkSession
from datetime import datetime, date, timedelta
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import random
import pandas as pd
import numpy as np


def label_transactions(venmo_df, word_df, emoji_df, spark, save_data = False):

    ### create a view for dataframes
    venmo_df.createOrReplaceTempView('venmo')
    word_df.createOrReplaceTempView('word_df')
    emoji_df.createOrReplaceTempView('emoji_df')

    matched_df = spark.sql("""
    SELECT temp.*, 
        CASE
            WHEN temp.emoji_label IS NULL AND temp.word_label IS NOT NULL THEN 'word only'
            WHEN temp.emoji_label IS NOT NULL AND temp.word_label IS NULL THEN 'emoji only'
            ELSE 'both emoji and word'
        END AS match_type
    FROM (
        SELECT s.story_id , FIRST(s.user1) AS user1 , FIRST(s.user2) AS user2 , FIRST(s.transaction_type) AS transactioni_type ,
        FIRST(s.description) AS description , FIRST(s.datetime) AS transaction_time, FIRST(e.label) AS emoji_label, FIRST(w.label) AS word_label,
        FIRST(e.token) AS emoji_token, FIRST(w.token) AS word_token
        FROM venmo s
        LEFT JOIN emoji_df e ON INSTR(s.description, e.token) > 0
        LEFT JOIN word_df w ON INSTR(s.description, CONCAT(';', w.token, ';')) > 0
        WHERE e.token IS NOT NULL OR w.token IS NOT NULL
        GROUP BY s.story_id
    ) temp
    """)

    matched_df = matched_df.withColumn("selected_match_type", when(col("emoji_label").isNull(), col("word_label")).otherwise(col("emoji_label")))

    if save_data == True:
        matched_df.write.format("parquet").option("compression", "gzip").save("./output/matched_df.parquet")

    return matched_df
