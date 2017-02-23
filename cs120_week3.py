#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:08:30 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime
import numpy as np
from pyspark.ml.linalg import SparseVector 
from pyspark.sql.functions import udf , explode , split ,col ,mean , sum
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, StructType, StructField, LongType, StringType ,DoubleType
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import when, log, col
from pyspark.sql.functions import lit
from math import exp
from collections import defaultdict
import hashlib


epsilon = 1e-16

sc = SparkContext("local[4]","cs120 week3")
sc.setLogLevel("ERROR")
sqc = SQLContext(sc)

def Log(*context):
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
    Log("Free Memory is ",memNotUsed_G , "G")
    return memNotUsed_G 


def clear_env():
    '''
    for prevent memory error,clear python veriable
    '''
    for key in globals().keys():
        if key not in ["memory_stat" , "Log" , "time" , "datetime"]:
            if not key.startswith("__"):
                globals().pop(key)
    memory_stat()
    Log("The python env is cleared")
    memory_stat()


try:
    #OHE
    from collections import defaultdict
    sample_one = [(0, 'mouse'), (1, 'black')]
    sample_two = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
    sample_three =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]
    def sample_to_row(sample):
        tempDic = defaultdict(lambda : None)
        tempDic.update(sample)
        return [tempDic[i] for i in range(3)]
    a = map(sample_to_row, [sample_one, sample_two, sample_three])
    Log(a)
    #load data
    raw_df = (sqc.read
              .text('./data/ctr_small.txt')
              .withColumnRenamed('value','text')
              )
    #raw_df = raw_df_t.select(split(raw_df_t.text,'\t'))
    raw_df.printSchema()
    raw_df.show(truncate = False)
    
    def one_hot_encoding(raw_feats, ohe_dict_broadcast, num_ohe_feats):
        '''
        Args:
        raw_feats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sample_one)
        ohe_dict_broadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.
        num_ohe_feats (int): The total number of unique OHE features (combinations of featureID and
            value).

        Returns:
            SparseVector: A SparseVector of length num_ohe_feats with indices equal to the unique
                identifiers for the (featureID, value) combinations that occur in the observation and
                with values equal to 1.0.
        '''
        for reat in sorted(raw_feats):
          if reat not in ohe_dict_broadcast.value:
            ohe_dict_broadcast.value[reat] = max(ohe_dict_broadcast.value.values()) + 1
        result = np.zeros(len(ohe_dict_broadcast.value))
        for reat in raw_feats:
          result[ohe_dict_broadcast.value[reat]] = 1.0
        return SparseVector(len(result),np.where(result>0)[0],result[np.where(result>0)])
        
    def ohe_udf_generator(ohe_dict_broadcast):
        """Generate a UDF that is setup to one-hot-encode rows with the given dictionary.
    
        Note:
            We'll reuse this function to generate a UDF that can one-hot-encode rows based on a
            one-hot-encoding dictionary built from the training data.  Also, you should calculate
            the number of features before calling the one_hot_encoding function.
    
        Args:
            ohe_dict_broadcast (Broadcast of dict): Broadcast variable containing a dict that maps
                (featureID, value) to unique integer.
    
        Returns:
            UserDefinedFunction: A UDF can be used in `DataFrame` `select` statement to call a
                function on each row in a given column.  This UDF should call the one_hot_encoding
                function with the appropriate parameters.
        """
        length = len(ohe_dict_broadcast.value)
        return udf(lambda x: one_hot_encoding(x ,ohe_dict_broadcast,length ), VectorUDT())

    def create_one_hot_dict(input_df):
        """Creates a one-hot-encoder dictionary based on the input data.
    
        Args:
            input_df (DataFrame with 'features' column): A DataFrame where each row contains a list of
                (featureID, value) tuples.
    
        Returns:
            dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
                unique integers.
        """
        return input_df.select(explode(input_df.features)).distinct().rdd.map(lambda x:tuple(x[0])).zipWithIndex().collectAsMap()
    
    weights = [.8, .1, .1]
    seed = 42
    
    # Use randomSplit with weights and seed
    raw_train_df, raw_validation_df, raw_test_df = raw_df.randomSplit(weights,seed)
    raw_train_df.cache()
    raw_validation_df.cache()
    raw_test_df.cache()
    Log("train dataset count is :" , raw_train_df.count())
    Log("val dataset count is :" , raw_validation_df.count())
    Log("test dataset count is :" , raw_test_df.count())
    
    def parse_point(point):
        """Converts a \t separated string into a list of (featureID, value) tuples.
    
        Note:
            featureIDs should start at 0 and increase to the number of features - 1.
    
        Args:
            point (str): A comma separated string where the first value is the label and the rest
                are features.
    
        Returns:
            list: A list of (featureID, value) tuples.
        """
        splitList = point.replace('\n','').split('\t')
        returnData = []
        for i in range(1,len(splitList)):
            returnData.append((i-1,splitList[i]))
        return returnData
    
    parse_point_udf = udf(parse_point, ArrayType(StructType([StructField('_1', LongType()),
                                                         StructField('_2', StringType())])))
    def parse_raw_df(raw_df):
        """Convert a DataFrame consisting of rows of comma separated text into labels and feature.
    
    
        Args:
            raw_df (DataFrame with a 'text' column): DataFrame containing the raw comma separated data.
    
        Returns:
            DataFrame: A DataFrame with 'label' and 'feature' columns.
        """
        #return sqc.createDataFrame(raw_df.rdd.map(lambda x : (float(x[0]) , parse_point_udf(x))),['label','feature'])
        return raw_df.select(split(raw_df.text,'\t').getItem(0).cast('double').alias('label') , parse_point_udf(raw_df.text).alias('feature'))
    parsed_raw_df = parse_raw_df(raw_df)
    parsed_train_df = parse_raw_df(raw_train_df)
    parsed_train_df.printSchema()
    parsed_train_df.show(2,truncate = False)
    parsed_validation_df = parse_raw_df(raw_validation_df)
    parsed_test_df = parse_raw_df(raw_test_df)
    parsed_test_df.cache()
    parsed_validation_df.cache()
    
    num_categories = (parsed_train_df
                    .select(explode('feature').alias('feature'))
                    .distinct()
                    .select(col('feature').getField('_1').alias('featureNumber'))
                    .groupBy('featureNumber')
                    .sum()
                    .orderBy('featureNumber')
                    .collect())
    Log("num_categories are " ,num_categories)
    
    ctr_ohe_dict = create_one_hot_dict(parsed_raw_df.select(parsed_raw_df.feature.alias('features')))
    #Log('ctr ohe dict are :' , ctr_ohe_dict)
    
    ohe_dict_broadcast = sc.broadcast(ctr_ohe_dict)
    ohe_dict_udf = ohe_udf_generator(ohe_dict_broadcast)
    ohe_train_df = parsed_train_df.select(parsed_train_df.label.alias('label'),ohe_dict_udf(parsed_train_df.feature).alias('feature'))
    ohe_train_df.cache()
    ohe_validation_df = parsed_validation_df.select(parsed_validation_df.label.alias('label'),ohe_dict_udf(parsed_validation_df.feature).alias('feature'))
    ohe_test_df = parsed_test_df.select(parsed_test_df.label.alias('label'),ohe_dict_udf(parsed_test_df.feature).alias('feature'))
    ohe_validation_df.cache()
    ohe_test_df.cache()
    ohe_train_df.printSchema()

    #print parsed_train_df.take(1)
    #print ohe_train_df.take(1)
    
    lr = LogisticRegression(featuresCol="feature",
                            labelCol="label" ,
                            maxIter=20 ,
                            standardization=False ,
                            regParam=0.01, 
                            elasticNetParam=0.0)
    lr_model_basic = lr.fit(ohe_train_df)
    Log("lr model intercept are" , lr_model_basic.intercept)
    Log("lr model coefficients are" , lr_model_basic.coefficients)
    
    
    def add_log_loss(df):
        """Computes and adds a 'log_loss' column to a DataFrame using 'p' and 'label' columns.
    
        Note:
            log(0) is undefined, so when p is 0 we add a small value (epsilon) to it and when
            p is 1 we subtract a small value (epsilon) from it.
    
        Args:
            df (DataFrame with 'p' and 'label' columns): A DataFrame with a probability column
                'p' and a 'label' column that corresponds to y in the log loss formula.
    
        Returns:
            DataFrame: A new DataFrame with an additional column called 'log_loss' where 'log_loss' column contains the loss value as explained above.
        """
        return (df.select(when(df.p==0,epsilon).otherwise(df.p).alias('p') , 'label')
                .select(when(col('p')==1,col('p')-epsilon).otherwise(col('p')).alias('p') , 'label')
                .select('*',when(col('label') == 1 , -1*log(col('p'))).otherwise(-1*log(1-col('p'))).alias('log_loss')))


    class_one_frac_train = ohe_train_df.select(mean(ohe_train_df.label)).collect()[0][0]
    log_loss_tr_base = add_log_loss(ohe_train_df.select(ohe_train_df.label.alias('label') , lit(class_one_frac_train).alias('p')))
    
    baseLogLossSum = log_loss_tr_base.select(mean(log_loss_tr_base.log_loss)).collect()[0][0]
    Log("Base line model avg of log loss is " , baseLogLossSum)
    Log("Base line model p is ",class_one_frac_train)
    
    
    def add_probability(df, model):
        """Adds a probability column ('p') to a DataFrame given a model"""
        coefficients_broadcast = sc.broadcast(model.coefficients)
        intercept = model.intercept
    
        def get_p(features):
            """Calculate the probability for an observation given a list of features.
    
            Note:
                We'll bound our raw prediction between 20 and -20 for numerical purposes.
    
            Args:
                features: the features
    
            Returns:
                float: A probability between 0 and 1.
            """
            # Compute the raw value
            raw_prediction = 1/(1+exp(-1*(features.dot(coefficients_broadcast.value)+intercept)))
            # Bound the raw value between 20 and -20
            #raw_prediction = <FILL IN>
            # Return the probability
            return raw_prediction
    
        get_p_udf = udf(get_p, DoubleType())
        return df.withColumn('p', get_p_udf('feature'))

    add_probability_model_basic = lambda df: add_probability(df, lr_model_basic)
    training_predictions = add_probability_model_basic(ohe_train_df).cache()
    training_predictions.show(3)
    
    def evaluate_results(df, model, baseline=None):
        """Calculates the log loss for the data given the model.
    
        Note:
            If baseline has a value the probability should be set to baseline before
            the log loss is calculated.  Otherwise, use add_probability to add the
            appropriate probabilities to the DataFrame.
    
        Args:
            df (DataFrame with 'label' and 'features' columns): A DataFrame containing
                labels and features.
            model (LogisticRegressionModel): A trained logistic regression model. This
                can be None if baseline is set.
            baseline (float): A baseline probability to use for the log loss calculation.
    
        Returns:
            float: Log loss for the data.
        """
        with_probability_df = add_probability(df , model)
        with_log_loss_df = add_log_loss(with_probability_df)
        log_loss = with_log_loss_df.select(mean(with_log_loss_df.log_loss)).collect()[0][0]
        return log_loss
    
    trainDataLogLoss = evaluate_results(ohe_train_df , lr_model_basic)
    Log("trainDataLogLoss is " ,trainDataLogLoss)
    testDataLogLoss = evaluate_results(ohe_test_df , lr_model_basic)
    Log("testDataLogLoss is " ,testDataLogLoss)
    
    #draw ROC curve
    import matplotlib.pyplot as plt
    def prepare_plot(xticks, yticks, figsize=(10.5, 6), hide_labels=False, grid_color='#999999',
                 grid_width=1.0):
        """Template for generating the plot layout."""
        plt.close()
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hide_labels: axis.set_ticklabels([])
        plt.grid(color=grid_color, linewidth=grid_width, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        return fig, ax
    
    labels_and_scores = add_probability_model_basic(ohe_validation_df).select('label', 'p')
    labels_and_weights = labels_and_scores.collect()
    labels_and_weights.sort(key=lambda (k, v): v, reverse=True)
    #every point of ROC curve depand on the threshold of p which is sorted by Labels_and_weights
    #points larger than p are all labeled by postive which calc TP and FP ,than calc TPR and FPR 
    labels_by_weight = np.array([k for (k, v) in labels_and_weights])
    
    length = labels_by_weight.size
    true_positives = labels_by_weight.cumsum()
    num_positive = true_positives[-1]
    false_positives = np.arange(1.0, length + 1, 1.) - true_positives
    
    true_positive_rate = true_positives / num_positive
    false_positive_rate = false_positives / (length - num_positive)
    
    # Generate layout and plot data
    fig, ax = prepare_plot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
    ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    plt.plot(false_positive_rate, true_positive_rate, color='#8cbfd0', linestyle='-', linewidth=3.)
    plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
    plt.show()
    #reduce feature dimension
    Log('The dimension of base model which without hash  is ',len(ctr_ohe_dict))
    Log('The dimension of hash model is ' , 2**15)
    def hash_function(raw_feats, num_buckets, print_mapping=False):
        """Calculate a feature dictionary for an observation's features based on hashing.
    
        Note:
            Use print_mapping=True for debug purposes and to better understand how the hashing works.
    
        Args:
            raw_feats (list of (int, str)): A list of features for an observation.  Represented as
                (featureID, value) tuples.
            num_buckets (int): Number of buckets to use as features.
            print_mapping (bool, optional): If true, the mappings of featureString to index will be
                printed.
    
        Returns:
            dict of int to float:  The keys will be integers which represent the buckets that the
                features have been hashed to.  The value for a given key will contain the count of the
                (featureID, value) tuples that have hashed to that key.
        """
        mapping = { category + ':' + str(ind) : int(int(hashlib.md5(category + ':' + str(ind)).hexdigest(), 16) % num_buckets) for ind , category in raw_feats}
        if print_mapping:
            print mapping
        def map_update(l,r):
            l[r] += 1.0
            return l
        sparse_features = reduce(map_update , mapping.values() , defaultdict(float))
        return dict(sparse_features)
    
    num_hash_buckets = 2**15
    tuples_to_hash_features_udf = udf(lambda x : Vectors.sparse(num_hash_buckets ,hash_function(x , num_hash_buckets) ), VectorUDT())
    def add_hash_features(df):
        return df.select(df.label.alias('label'),tuples_to_hash_features_udf(df.feature).alias('feature'))
    hash_train_df = add_hash_features(parsed_train_df)
    hash_validation_df = add_hash_features(parsed_validation_df)
    hash_test_df = add_hash_features(parsed_test_df)
    hash_train_df.cache()
    hash_validation_df.cache()
    hash_test_df.cache()
    
    Log("Compare hash_train_df and ohe_train_df")
    hash_train_df.show()
    ohe_train_df.show()
    
    def vector_feature_sparsity(sparse_vector):
        """Calculates the sparsity of a SparseVector.
    
        Args:
            sparse_vector (SparseVector): The vector containing the features.
    
        Returns:
            float: The ratio of features found in the vector to the total number of features.
        """
        return len(sparse_vector.indices)*1.0 / len(sparse_vector)
    feature_sparsity_udf = udf(vector_feature_sparsity, DoubleType())
    
    def get_sparsity(df):
        """Calculates the average sparsity for the features in a DataFrame.
    
        Args:
            df (DataFrame with 'features' column): A DataFrame with sparse features.
    
        Returns:
            float: The average feature sparsity.
        """
        return df.select(feature_sparsity_udf(df.feature).alias('sparsity')).select(mean('sparsity')).collect()[0][0]
    
    average_sparsity_ohe = get_sparsity(ohe_train_df)
    average_sparsity_hash = get_sparsity(hash_train_df) 
    Log("average sparsity ohe : " ,average_sparsity_ohe)
    Log("average sparsity hash :" , average_sparsity_hash)
    
    lr_hash = LogisticRegression(featuresCol="feature",
                            labelCol="label" ,
                            maxIter=20 ,
                            standardization=False ,
                            regParam=0.01, 
                            elasticNetParam=0.7)
    lr_model_hashed = lr_hash.fit(hash_train_df)
    log_loss_train_model_hashed = evaluate_results(hash_train_df , lr_model_hashed)
    Log("intercept of hashed model are " , lr_model_hashed.intercept)
    Log("The log loss of hashed model is " , log_loss_train_model_hashed)
    Log("The log loss of ohe model is " , trainDataLogLoss)
    
    log_loss_test = evaluate_results(hash_test_df , lr_model_hashed)
    Log("The log loss of test hash df is " , log_loss_test)
    Log("The log loss of test ohe df is " , testDataLogLoss)
    
    class_one_frac_test = hash_test_df.select(mean(hash_test_df.label)).collect()[0][0]
    log_loss_test_base = add_log_loss(hash_test_df.select(hash_test_df.label.alias('label') , lit(class_one_frac_train).alias('p')))
    baseLogLosstest = log_loss_test_base.select(mean(log_loss_test_base.log_loss)).collect()[0][0]
    Log("Base line model avg of log loss is " , baseLogLosstest)
    
    
except Exception as e:
    Log("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    Log("[INFO]The spark cluster stopped")
    clear_env()