#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:01:39 2017

@author: hp
"""

from pyspark import SparkContext
#from pyspark.sql.context import HiveContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
import pyspark.sql.functions as pyfunc
from pyspark.sql.types import BooleanType
import traceback
import time

sc = SparkContext("local[2]","cs105Lecture App")
sc.setLogLevel("ERROR")
#sqc = HiveContext(sc) #meet error :Another instance of Derby 
                      #may have already booted the database
sqc = SQLContext(sc)

def outPutPrint(*context):
    '''
    for output some infomation
    '''
    outputlogo = "---->" + "[" + time.ctime() + "]"
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
        if key not in ["memory_stat" , "outPutPrint" , "time"]:
            if not key.startswith("__"):
                globals().pop(key)
    memory_stat()
    outPutPrint("The python env is cleared")
    memory_stat()


outPutPrint("[INFO] the spark cluster started")
outPutPrint(type(sqc))
outPutPrint(sc.version)


try:
    #spark sql join
    data1 = [('Alice', 1), ('Bob', 2), ('Bill', 4)]
    df1 = sqc.createDataFrame(data1,['name','age'])
    data2 = [('jia', 165), ('Bob', 150), ('Bill', 157)]
    df2 = sqc.createDataFrame(data2,['name','heigh'])
    
    df4 = df1.join(df2,'name' , "inner")
    df4.show()
    df4.select(df4.name , df4.age).show()
    
    jointype = ['inner', 'outer', 'left_outer', 'right_outer', 'leftsemi']
    for t in jointype:
        outPutPrint("The join type is ",t)
        df5 = df1.join(df2 , 'name' , t)
        df5.show()
        
    #cs105-lab1a
    from faker import Factory
    fake = Factory.create()
    fake.seed(4321)
    
    def fake_entry():
        name = fake.name().split()
        return (name[1] , name[0] ,fake.ssn() , fake.job() , abs(2016-fake.date_time().year)+1)
    def repeat(times , func , *args , **kwargs):
        for _ in xrange(times):
            yield func(*args , **kwargs)
            
    data = list(repeat(10000 ,fake_entry))
    dataDF = sqc.createDataFrame(data,('last_name', 'first_name', 'ssn', 'occupation', 'age'))
    outPutPrint("type of dataDf ",type(dataDF))
    dataDF.printSchema()
    dataDF.cache()
    sqc.registerDataFrameAsTable(dataDF , 'dataDF_T')
    rdd_num = dataDF.rdd.getNumPartitions()
    outPutPrint("dataDF's partitions num is : " , rdd_num)
    newDF = dataDF.select("*")
    newDF.explain(True)
    subDF = dataDF.select('last_name', 'first_name', 'ssn' ,'occupation',  (dataDF.age -1).alias('age'))
    subDF.show(n = 30 ,truncate = True)
    outPutPrint("count dataDF" , dataDF.count())
    outPutPrint("count subDF" , subDF.count())
    filterDF = subDF.filter(subDF.age < 10)
    filterDF.explain(True)
    filterDF.show()
    outPutPrint("count filterDF ",filterDF.count())
    
    lessThen = pyfunc.udf(lambda s: s < 10 , BooleanType())
    lambdaDF = subDF.filter(lessThen(subDF.age))
    lambdaDF.show()
    outPutPrint("count lambdaDF",lambdaDF.count())
    
    outPutPrint("first element of lambdaDF" , lambdaDF.first())
    outPutPrint("some elements of lambdaDF" , lambdaDF.take(10))
    
    dataDF.orderBy(dataDF.age.desc()).show()
    
    outPutPrint("dataDF distinct count is ", dataDF.distinct().count())
    
    tempDF = sqc.createDataFrame([("Joe", 1), ("Joe", 1), ("Anna", 15), ("Anna", 12), ("Ravi", 5)], ('name', 'score'))
    tempDF.show()
    tempDF.distinct().show()
    
    outPutPrint("dataDF count of name col ",dataDF.select(dataDF.last_name , dataDF.first_name).distinct().count())
    outPutPrint("use dropduplicates on dataDF",dataDF.dropDuplicates(['last_name','first_name']).count())
    outPutPrint("dataDF count ", dataDF.count())
    
    dataDF.drop('age').drop('occupation').printSchema()
    dataDF.printSchema()
    
    groupDF = dataDF.groupBy('occupation').count()
    groupDF.show()
    groupDF.printSchema()
    
    dataDF.groupBy().avg('age').show()
    
    firstMaxElem = (dataDF
                        .groupBy()
                        .max('age')
                        .first() )
    outPutPrint("first return type" , type(firstMaxElem) ,
                len(firstMaxElem) , type(firstMaxElem[0]) ,
                   firstMaxElem[0])
    
    sampleDF = dataDF.sample(withReplacement = False , fraction = 0.1)
    outPutPrint("sample 10% of dataDF ,the num of sample is ",
                sampleDF.count())
    sampleDF.show()
    
    outPutPrint("is dataDF cached ?" , dataDF.is_cached)
    dataDF.unpersist()
    outPutPrint("is dataDF cached ?" , dataDF.is_cached)
    

except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()