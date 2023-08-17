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


def data_preprocess(df, dictionary_list, spark): ### put the venmo df and the mapping dictionaries
    ### data preprocessing for venmo transactions
    def df_preprocess(df):
        ### standardize text format
        df = df.withColumn('description' , lower(df['description']))

        ### replace every white space with ";" and insert ";" to the start and beginning
        df = df.withColumn("description", regexp_replace(concat(lit(";"),\
                                    df["description"] , lit(";")) , " " , ";"))
        ### create time
        df = df.withColumn('time', date_format('datetime', 'HH:mm:ss'))

        return df


    def create_mapping_pyspark_table(dictionary_list): ### melt dataframe into two columns and return spark dataframe
        ### melt dataframe and convert to two columns
        for dict in range(len(dictionary_list)):
            ### remove extra white space in column names
            dictionary_list[dict].columns = [col.strip(" ") for col in dictionary_list[dict].columns]
            dictionary_list[dict] = dictionary_list[dict].melt().dropna(subset = ['variable' , 'value'])
            
            ### identify the source of mapping table
            if dict == 0:
                dictionary_list[dict]['source'] = 'emoji_dict'
            else:
                dictionary_list[dict]['source'] = 'word_dict'
        
        ### rename columns
        new_df = pd.concat(dictionary_list).rename(columns = {'variable':'label', 'value':'token'})

        ### remove unname values
        for col in new_df.columns:
            filter = new_df[col].str.contains('.*Unname.*')
            new_df = new_df[~filter]

        return spark.createDataFrame(new_df)
    
    venmo_df, mapping_df = df_preprocess(df), create_mapping_pyspark_table(dictionary_list)
    emoji_df = mapping_df[mapping_df['source'] == 'emoji_dict']
    word_df = mapping_df[mapping_df['source'] == 'word_dict']

    return venmo_df, emoji_df, word_df