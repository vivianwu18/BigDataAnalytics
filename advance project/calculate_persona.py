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

def calculate_community_persona(df , start_lifetime_month = 1, end_lifetime_month = 12):
    ### only select the friend's transactions within user lifetime
    window_spec = Window.partitionBy('user').orderBy('transaction_time')
    df = df.withColumn('user_lifetime_for_friend_transactions', ((datediff('friend_transaction_time', first('transaction_time').over(window_spec))) / 30.44 + 1) .cast(IntegerType()))
    df = df.filter((col('lifetime_month') >= start_lifetime_month) & (col('lifetime_month') <= end_lifetime_month)) ### start lifetime month & end lifetime month

    ### count the transactions of user's community
    pivot_df = df.select('user', 'friend', 'lifetime_month', 'friend_selected_match_type')
    pivot_df = (
        df.groupBy('user', 'friend', 'lifetime_month')
        .pivot('friend_selected_match_type')
        .agg(count('*')))

    window_spec = Window.partitionBy('user', 'friend').orderBy('lifetime_month').rowsBetween(Window.unboundedPreceding, 0)

    ### calculate the accumulated count for each selected match type
    accumulated_cols = [c for c in pivot_df.columns if c not in ['user', 'friend', 'lifetime_month']]
    for col_name in accumulated_cols:
        pivot_df = pivot_df.withColumn(
            col_name,
            sum(col(col_name)).over(window_spec).alias(col_name)
        )

    ### fill null values with 0 
    pivot_df = pivot_df.fillna(0)

    ### count the total transactions
    friend_trans_total_counts = pivot_df.withColumn('total_count', 
        col('Activity') + col('Cash') + col('Event') + col('Food') + col('Illegal/Sarcasm') +
        col('People') + col('Transportation') + col('Travel') + col('Utility'))

    ### rename the columns
    friend_trans_total_counts = friend_trans_total_counts.withColumnRenamed('Activity', 'c_Activity') \
        .withColumnRenamed('Cash', 'c_Cash') \
        .withColumnRenamed('Event', 'c_Event') \
        .withColumnRenamed('Food', 'c_Food') \
        .withColumnRenamed('Illegal/Sarcasm', 'c_Illegal/Sarcasm') \
        .withColumnRenamed('People', 'c_People') \
        .withColumnRenamed('Transportation', 'c_Transportation') \
        .withColumnRenamed('Travel', 'c_Travel') \
        .withColumnRenamed('Utility', 'c_Utility')

    ### calculate the percentage
    for col_name in friend_trans_total_counts.columns[3:]:
        friend_trans_total_counts = friend_trans_total_counts.withColumn(col_name , col(col_name) / col('total_count'))
        
    friend_trans_total_counts = friend_trans_total_counts.drop('total_count') ### drop unnecessary columns

    ### calculate community spending profile
    community_df = friend_trans_total_counts.drop('friend')

    ### calculate the monthly average
    average_window_spec = Window.partitionBy('user', 'lifetime_month').orderBy('lifetime_month')

    for col_name in community_df.columns[2:]:
        community_df = community_df.withColumn(col_name, avg(col(col_name)).over(average_window_spec).alias(col_name))

    ### calculate the rolling monthly average
    community_df = community_df.distinct()
    rolling_window_spec = Window.partitionBy('user').orderBy('lifetime_month')

    for col_name in community_df.columns[2:]:
        community_df = community_df.withColumn(col_name, avg(col(col_name)).over(rolling_window_spec).alias(col_name))

    ### filter the specific lifetime month
    grouped_df = community_df.groupBy('user').agg(max('lifetime_month').alias('max_month'))

    return_df = community_df.alias('c').join(grouped_df.alias('g'),
                                        (col('c.user') == col('g.user')) &
                                        ((col('c.lifetime_month') == end_lifetime_month) |
                                        (col('c.lifetime_month') == col('g.max_month'))),
                                        'inner') \
                            .select('c.*')
    return_df = return_df.drop('lifetime_month')

    return return_df

def calculate_self_persona(df, end_lifetime_month):
    pivot_df = (
        df.groupBy('user', 'lifetime_month')
        .pivot('selected_match_type')
        .agg(count('*'))
    )

    window_spec = Window.partitionBy('user').orderBy('lifetime_month').rowsBetween(Window.unboundedPreceding, 0)
    accumulated_cols = [c for c in pivot_df.columns if c not in ['user', 'lifetime_month']]
    for col_name in accumulated_cols:
        pivot_df = pivot_df.withColumn(
            col_name,
            sum(col(col_name)).over(window_spec).alias(col_name)
        )

    ### fill null values with 0 in the selected columns
    pivot_df = pivot_df.fillna(0)

    ### count the total transactions
    self_df = pivot_df.withColumn('total_count', col('Activity') + col('Cash') +col('Event') +col('Food') + col('Illegal/Sarcasm') +  col('People') + col('Transportation') +\
                                                col('Travel') + col('Utility'))

    ### calculate the percentage
    for col_name in self_df.columns[2:]:
        self_df = self_df.withColumn(col_name , col(col_name) / col('total_count'))
        
    ### filter the specific lifetime month
    self_df = self_df.drop('total_count')
    grouped_df = self_df.groupBy('user').agg(max('lifetime_month').alias('max_month'))

    return_df = self_df.alias('s').join(grouped_df.alias('g'),
                                        (col('s.user') == col('g.user')) &
                                        ((col('s.lifetime_month') == end_lifetime_month) |
                                        (col('s.lifetime_month') == col('g.max_month'))),
                                        'inner') \
                            .select('s.*')
    return_df = return_df.drop('lifetime_month')
    
    return return_df

def calculate_persona(df, spark, start_lifetime_month = 1, end_lifetime_month = 12, save_data = False):
    
    print("stage : start user combine.......", end="")
    matched_df_all = df.select('user1', 'user2', 'transaction_time', 'selected_match_type')\
    .withColumnRenamed('user1' , 'user')\
    .withColumnRenamed('user2' , 'friend')\
    .union(df.select('user2', 'user1' , 'transaction_time', 'selected_match_type')\
            .withColumnRenamed('user2' , 'user')\
            .withColumnRenamed('user1' , 'friend'))
    print("complete")
    
    print("stage : start calculating lifetime month.......", end="")    
    ### calculate lifetime month for users
    window_spec = Window.partitionBy('user').orderBy('transaction_time')
    matched_df_all = matched_df_all.withColumn('lifetime_month', ((datediff('transaction_time', first('transaction_time').over(window_spec))) / 30.44 + 1) .cast(IntegerType()))
    print("complete")
    
    ### make DataFrame into a view
    matched_df_all.createOrReplaceTempView('matched_df_all')
    
    print("stage : start joining friends transactions.......", end="") 
    ### join friend's transactions
    combined_df = spark.sql("""
        SELECT m1.*, m2.transaction_time AS friend_transaction_time, m2.selected_match_type AS friend_selected_match_type
        FROM matched_df_all AS m1
        JOIN matched_df_all AS m2 ON m1.friend = m2.user AND m2.friend != m1.user
        JOIN (
            SELECT DISTINCT user, friend, MIN(transaction_time) AS earliest_transaction_time
            FROM matched_df_all
            GROUP BY user, friend
        ) AS m3 ON m1.user = m3.user AND m1.friend = m3.friend AND m1.transaction_time = m3.earliest_transaction_time
    """)
    print("complete")
    
    if start_lifetime_month != 1:
        print("stage : start calculating user persona.......", end="") 
        matched_df_all1 = matched_df_all.filter(col('lifetime_month') <= start_lifetime_month)
        persona_self1 = calculate_self_persona(matched_df_all1 , end_lifetime_month).orderBy('user')
        matched_df_all2 = matched_df_all.filter((col('lifetime_month') > start_lifetime_month) & (col('lifetime_month') <= end_lifetime_month))
        persona_self2 = calculate_self_persona(matched_df_all2 , end_lifetime_month).orderBy('user')
        persona_self2 = persona_self2.select([col(col_name).alias('c_' + col_name) for col_name in persona_self2.columns])
        print('complete')

        final_df = persona_self1.join(persona_self2, on = (persona_self1.user == persona_self2.c_user), how = 'left')
        final_df = final_df.drop('c_user')
        final_df = final_df.fillna(0)

        ### save the output
        if save_data == True:
            final_df.write.format("parquet").option("compression", "gzip").save("./output/final_df.parquet")
            
        return final_df

    else:
        ### calculate persona for both friends and user
        print("stage : start calculating user persona.......", end="") 
        matched_df_all1 = matched_df_all.filter((col('lifetime_month') >= start_lifetime_month) & (col('lifetime_month') <= end_lifetime_month))
        persona_self = calculate_self_persona(matched_df_all1 , end_lifetime_month).orderBy('user')
        print('complete')
        print("stage : start calculating friends persona.......", end="") 
        persona_community = calculate_community_persona(combined_df , start_lifetime_month, end_lifetime_month).orderBy('user')
        print('complete')

        final_df = persona_self.join(persona_community, on = 'user', how = 'left')
        final_df = final_df.fillna(0)

        ### save the output
        if save_data == True:
            final_df.write.format("parquet").option("compression", "gzip").save("./output/final_df.parquet")
            
        return final_df
