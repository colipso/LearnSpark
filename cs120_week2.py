#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:51:04 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.linalg import DenseVector
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression

sc = SparkContext("local[4]","cs120 week2")
sc.setLogLevel("ERROR")
sqc = SQLContext(sc)


def outPutPrint(*context):
    '''
    for output some infomation
    '''
    outputlogo = "---->" + "[" + str(datetime.datetime.now()) + "]"
    string_print = ""
    for c in context:
        string_print += str(c)+"  "
    content = outputlogo +string_print + '\n'
    f = open("log.txt",'a')
    f.write(content)
    f.close;
    print outputlogo,string_print,'\n'
    return True

def memory_stat():
    '''
    return memory usage. returntype is dict
    '''
    mem = {}  
    f = open("/proc/meminfo")  
    lines = f.readlines()  
    f.close()  
    for line in lines:  
        if len(line) < 2: continue  
        name = line.split(':')[0]  
        var = line.split(':')[1].split()[0]  
        mem[name] = long(var) * 1024.0  
    #mem['MemUsed'] = mem['MemTotal'] - mem['MemFree'] - mem['Buffers'] - mem['Cached']
    memNotUsed_G = "%.2f"% (mem['MemFree'] *1.0/(1024.0*1024.0*1024.0))
    outPutPrint("Free Memory is ",memNotUsed_G , "G")
    return memNotUsed_G 


def clear_env():
    '''
    for prevent memory error,clear python veriable
    '''
    for key in globals().keys():
        if key not in ["memory_stat" , "outPutPrint" , "time" , "datetime"]:
            if not key.startswith("__"):
                globals().pop(key)
    memory_stat()
    outPutPrint("The python env is cleared")
    memory_stat()


try:
    #load data
    raw_data_df = sc.textFile('./data/YearPredictionMSD.txt')
    def str2float(row):
        dataList = row.split(',')
        returnData = []
        for data in dataList:
            returnData.append(float(data))
        return LabeledPoint(returnData[0] , returnData[1:])
    
    parsed_points_df = raw_data_df.map(str2float).toDF()
    print parsed_points_df.count()
    print parsed_points_df.take(1)
    parsed_points_df.printSchema()
    
    #observe data
    content_stats_min = parsed_points_df.select(F.min(parsed_points_df.label)).collect()
    print content_stats_min
    content_stats_max = parsed_points_df.select(F.max(parsed_points_df.label)).collect()
    print content_stats_max
    
    #split data
    parsed_train_data_df, parsed_val_data_df, parsed_test_data_df = parsed_points_df.randomSplit([.8,.1,.1],42)
    parsed_train_data_df.cache() 
    parsed_val_data_df.cache()
    parsed_test_data_df.cache()
    
    average_train_year = parsed_train_data_df.selectExpr('avg(label)').first()[0]
    outPutPrint("average trained year is " , average_train_year)
    
    #evaluate result
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
    def calc_RMSE(dataset):
        return evaluator.evaluate(dataset)
    
    def gradient_summand(weights , lp):
        return DenseVector((np.array(weights).dot(lp.features) - lp.label ) * weights)
    
    def get_labeled_prediction(weights , observation):
        return (float(np.array(weights).dot(observation.features)) ,float(observation.label))
    
    def linreg_gradient_descent(train_data , num_iters):
        n = train_data.count()
        d = len(train_data.first().features)
        w = np.zeros(d)
        alpha = 1.0
        error_train = np.zeros(num_iters)
        for i in range(num_iters):
            preds_and_labels_train = train_data.map(lambda x : get_labeled_prediction(w,x))
            preds_and_labels_train_df = sqc.createDataFrame(preds_and_labels_train , ['prediction','label'])
            error_train[i] = calc_RMSE(preds_and_labels_train_df)
            gradient = train_data.map(lambda x : gradient_summand(w,x)).reduce(lambda x, y : x + y)
            alpha_i =  alpha / (n * np.sqrt(i+1))
            w -= alpha_i * gradient
        return w , error_train
    
    def changeMLB2ml(df):
        return sqc.createDataFrame(df.rdd.map(lambda x : (x.label , Vectors.dense(x.features.values))),['label','features'])
    '''
    #train model
    weights_LR , error_lit = linreg_gradient_descent(parsed_train_data_df.rdd,30)
    
    #use pyspark ml
    from pyspark.ml.regression import LinearRegression
    lin_reg = LinearRegression(maxIter=50, 
                               regParam=1e-1 ,
                               elasticNetParam=.2 ,
                               featuresCol="features", 
                               labelCol="label")
    first_model = lin_reg.fit(changeMLB2ml(parsed_train_data_df))
    coefficients = first_model.coefficients
    intercept_LR = first_model.intercept
    outPutPrint("The linear regression coefficient are ",coefficients)
    outPutPrint("The linear regression intercept is ", intercept_LR)
    '''
    #use pipeline
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import PolynomialExpansion
    
    ploynomial_expansion = PolynomialExpansion(degree = 2 ,inputCol="features", outputCol="polyFeatures")
    linear_regression = LinearRegression(maxIter=50, regParam=1e-10, elasticNetParam=.2,
                                     fitIntercept=True, featuresCol='polyFeatures')
    pipline = Pipeline(stages = [ploynomial_expansion , linear_regression])
    pipeline_model = pipline.fit(changeMLB2ml(parsed_train_data_df))
    prediction_df = pipeline_model.transform(changeMLB2ml(parsed_train_data_df))
    outPutPrint("The RMSE of pipeline model is ",calc_RMSE(prediction_df))
    
except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()