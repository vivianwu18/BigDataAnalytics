import findspark
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from datetime import datetime, date, timedelta
from dateutil import relativedelta
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import random
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import os
import time
del sum

import warnings
warnings.filterwarnings("ignore")


# preprocess community data and return data ready for train and test
def preprocess_community_model_data(df, spark , train_test_percent = 0.8 , random_seed = 0):
    # check column type and rename
    df = df.select(
        df["user"],
       df["Activity"].cast("double").alias("label_activity"),
       df["Cash"].cast("double").alias("label_Cash"),
       df["Event"].cast("double").alias("label_Event"),
       df["Food"].cast("double").alias("label_Food"),
       df["Illegal/Sarcasm"].cast("double").alias("label_Illegal/Sarcasm"),
       df["People"].cast("double").alias("label_People"),
       df["Transportation"].cast("double").alias("label_Transportation"),
       df["Travel"].cast("double").alias("label_Travel"),
       df["Utility"].cast("double").alias("label_Utility"),
       df["c_Activity"].cast("double"),
       df["c_Cash"].cast("double"),
       df["c_Event"].cast("double"),
       df["c_Food"].cast("double"),
       df["c_Illegal/Sarcasm"].cast("double"),
       df["c_People"].cast("double"),
       df["c_Transportation"].cast("double"),
       df["c_Travel"].cast("double"),
       df["c_Utility"].cast("double"))

    
    # assemble independant variables
    x_variables = VectorAssembler(inputCols = ["c_Activity","c_Cash","c_Event","c_Food","c_Illegal/Sarcasm","c_People","c_Transportation","c_Travel","c_Utility"], outputCol="features")
    # transform
    source_transformed = x_variables.transform(df)
    
    # train test split
    train , test = source_transformed.randomSplit([train_test_percent , 1 - train_test_percent] , seed = random_seed)
    
    return train , test


# function for regression training
def train_regression(train_df, spark) :
    models = [] # model container
    label_col = [col for col in train_df.columns if "label" in col]

    # train individual model for each dependent variable
    for col in label_col: # iterate through all y
        print(f"stage : start training {col}.......", end="")
        regressor = LinearRegression(featuresCol = "features" , labelCol = col , predictionCol = "prediction")
        model = regressor.fit(train_df)
        models.append(model)
        print("complete")
        
    # create column for df making
    column_list = [c for c in train_df.columns if "c_" in c]
    # store all coef into nested list format
    coef_list = [list(vec.coefficients) + [vec.intercept] for vec in models]

    # create dataframe
    result_df = pd.DataFrame(coef_list , columns = column_list + ['intercept'])
    # create index column
    result_df['Dependent variable'] = [c.replace('c_','') for c in train_df.columns if "c_" in c]
    result_df.set_index('Dependent variable' , inplace = True)

    return models , result_df


def make_forecast(models , x_test_df, spark):

    # make column
    y_col = [col.replace('c_' , '') for col in x_test_df.columns if "c_" in col]
    
    # return complete dataframe
    for model_index in range(len(models)): # iterate through all the models and combine the prediction columns
        predict = models[model_index].transform(x_test_df)
        if model_index == 0:
            y_list = predict.withColumnRenamed("prediction" , f"{y_col[model_index]}_predict")
        else:
            predict = predict.select('user','prediction').withColumnRenamed("prediction" , f"{y_col[model_index]}_predict")
            y_list = y_list.join(predict , on = "user" , how = "inner")

   
    return y_list


def calculate_rmse(df, spark):
    
    # get x columns
    independ_cols = [col for col in df.columns if "c_" in col]
    # get y columns
    predict_cols = [col for col in df.columns if "_predict" in col]
    # get compare columns
    col_to_compare = [col.replace('c_' , '') for col in independ_cols]
    
    
    # calculate difference square
    for variable_index in range(len(col_to_compare)):
        if variable_index == 0:
            calculate_df = df.withColumn(f'{col_to_compare[variable_index]}_diff_square' , (df[independ_cols[variable_index]] - df[predict_cols[variable_index]])**2)
        else:
            calculate_df = calculate_df.withColumn(f'{col_to_compare[variable_index]}_diff_square' , (df[independ_cols[variable_index]] - df[predict_cols[variable_index]])**2)
    
    # calculate sum of dif square
    calculate_df = calculate_df.withColumn("total_difference_square" , sum(calculate_df[col] for col in calculate_df.columns if "diff_square" in col))
    
    # calculate average of column
    rmse = calculate_df.select(avg("total_difference_square")).first()[0]
    
    return rmse ,calculate_df

def load_or_save_model(dt_v, spark, model_list = None, operation = "save"): #date version is for user to input today's date and model version
    date_version = dt_v
    
    if operation == "save":
        # create directroy
        try:
            os.mkdir('models')
            os.mkdir(f'./models/{date_version}')
        except:
            pass
        
        # save models
        for i, model in enumerate(model_list):
            model.save(f'./models/{date_version}/model_{i}.model')
        
        return "model_saved!"
    
    elif operation == "load":
        model_list = []
        for model_name in os.listdir(f'./models/{dt_v}'):
            model = LinearRegressionModel.load(f'./models/{dt_v}/{model_name}')
            model_list.append(model)
        print("load sucessfully!")
        return model_list
        
    else:
        return "operation not specified!"