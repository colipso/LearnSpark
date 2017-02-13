#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:41:07 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime
import seaborn as sbn
from pyspark.sql.types import StructType , DoubleType , IntegerType , StringType , StructField
from pyspark.sql.types import *

sc = SparkContext("local[4]","cs110 week2 APP")
sc.setLogLevel("ERROR")
sqc = SQLContext(sc)

isBigFile = True


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
    #import data from text files
    if isBigFile:
        bigPath = './data/'
        ratingsFile = bigPath + 'ratings.csv'
        moviesFile = bigPath + 'movies.csv'
    else:
        smallPath = './data/smallDataSet/'
        ratingsFile = smallPath + 'ratings.csv'
        moviesFile = smallPath + 'movies.csv'
        
    ratings_df_schema = StructType(
            [StructField('userId',IntegerType()),
             StructField('movieId',IntegerType()),
             StructField('rating',DoubleType())]
            )
    movies_df_schema = StructType(
            [StructField('ID',IntegerType()),
             StructField('title',StringType())]
            )
    ratings_df = sqc.read.csv(ratingsFile,
                              header = True ,
                              schema = ratings_df_schema)
    outPutPrint("ratings contains records:",ratings_df.count())
    ratings_df.show()
    ratings_df.printSchema()
    movies_df = sqc.read.csv(moviesFile,
                             header = True ,
                             schema = movies_df_schema)
    movies_df.show()
    movies_df.printSchema()
    ratings_df.cache()
    movies_df.cache()
    
    outPutPrint("There are %d movies" % movies_df.count())
    
    #verificate data
    assert movies_df.filter(movies_df.title == 'Toy Story (1995)').count() == 1 , "someting Wrong"
    
    #basic recommendation
    from pyspark.sql import functions as F
    movie_ids_with_avg_ratings_df = ( ratings_df
                                     .groupBy('movieId')
                                     .agg(F.count(ratings_df.rating).alias('count') , F.avg(ratings_df.rating).alias('average'))
    )
    movie_ids_with_avg_ratings_df.show()
    movie_ids_with_avg_ratings_df.printSchema()
    movie_names_with_avg_ratings_df = (movie_ids_with_avg_ratings_df
                                       .join(movies_df , movies_df.ID == movie_ids_with_avg_ratings_df.movieId)
                                       .drop('ID'))
    movie_names_with_avg_ratings_df.show()
    movie_names_with_avg_ratings_df.printSchema()
    
    #choose at leaest 500 review and sort
    movies_with_500_ratings_or_more = (movie_names_with_avg_ratings_df
                                       .filter(movie_names_with_avg_ratings_df['count']>=500)
                                       .sort('average' , ascending=False))
    movies_with_500_ratings_or_more.show(truncate = False)
    movies_with_500_ratings_or_more.printSchema()
    
    #collaborative filtering
    seed = 1800009193L
    (training_df , validation_df , test_df) = ratings_df.randomSplit( [0.6,0.2,0.2], seed)
    
    from pyspark.ml.recommendation import ALS
    als = ALS(maxIter = 10 , seed = seed , regParam = 0.1 , userCol="userId", itemCol="movieId", ratingCol="rating")
    als.setPredictionCol('prediction')
    from pyspark.ml.evaluation import RegressionEvaluator
    reg_eval = RegressionEvaluator(predictionCol = 'prediction',labelCol='rating',metricName = 'rmse')
    ranks = [4,8,12]
    errors = []
    models = []
    err = 0
    min_err = float('inf')
    best_rank = -1
    for rank in ranks:
        als.setRank(rank)
        model = als.fit(training_df)
        predict_df = model.transform(validation_df)
        predict_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
        error = reg_eval.evaluate(predict_ratings_df)
        outPutPrint("For %d rank the RMSE is %f" % (rank , error) )
        if error < min_err:
            min_err = error
            best_rank = err
            err += 1
        errors.append(error)
        models.append(model)

    als.setRank(ranks[best_rank])
    outPutPrint("The best model was traind with rank %d" % ranks[best_rank])
    mymodel = models[best_rank]
    
    #test with average training set rating
    avg_rating_df = training_df.groupBy().agg(F.avg(training_df.rating))
    training_avg_rating = avg_rating_df.collect()[0][0]
    outPutPrint("the average movie rating of trainingset is " , training_avg_rating)
    test_for_avg_df = test_df.withColumn('prediction' , F.lit(training_avg_rating))
    test_avg_RSME = reg_eval.evaluate(test_for_avg_df)
    outPutPrint("the rsme of  average rating is ",test_avg_RSME)
    
    #recommend movies for myself
    from pyspark.sql import Row
    my_rated_movies = [
            (0,858,5),
            (0,1221,5),
            (0 ,2571,5),
            (0 , 7153,5),
            (0 , 4993,5),
            (0 , 79132,5)
            ]
    my_ratings_df = sqc.createDataFrame(my_rated_movies ,  ['userId','movieId','rating'])
    training_with_my_ratings_df = training_df.unionAll(my_ratings_df)
    my_rating_model = als.fit(training_with_my_ratings_df)
    my_predict_df = my_rating_model.transform(test_df)
    predicted_test_my_ratings_df = my_predict_df.filter(my_predict_df.prediction != float('nan'))
    test_RMSE_my_ratings = reg_eval.evaluate(predicted_test_my_ratings_df)
    outPutPrint("The model had a RMSE " , test_RMSE_my_ratings)
    
    my_rated_movie_ids = [x[1] for x in my_rated_movies]
    not_rated_df = movies_df.filter(~movies_df.ID.isin(my_rated_movie_ids))
    my_unrated_movies_df = not_rated_df.select('*' ,F.lit(0).alias('userId')).withColumnRenamed('ID' , 'movieId')
    raw_predicted_ratings_df = my_rating_model.transform(my_unrated_movies_df)
    predicted_ratings_df = raw_predicted_ratings_df.filter(raw_predicted_ratings_df['prediction'] != float('nan'))
    
    predicted_with_counts_df = predicted_ratings_df.join(movie_names_with_avg_ratings_df , movie_names_with_avg_ratings_df.movieId == predicted_ratings_df.movieId)
    predicted_highest_rated_movies_df = predicted_with_counts_df.filter(predicted_with_counts_df['count'] > 100).sort(predicted_with_counts_df.prediction ,ascending = False)
    predicted_highest_rated_movies_df.show(20,truncate = False)
    
    
except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()
    