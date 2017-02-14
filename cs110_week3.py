#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:16:24 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime

sc = SparkContext("local[4]","cs110 week3 APP")
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
    #learn rdd
    import random
    data = []
    for i in xrange(10):
        data.append(random.random()*100)
    rdd = sc.parallelize(data , 4)
    outPutPrint("rdd is: " , rdd.collect())
    outPutPrint("double rdd is: " ,rdd.map(lambda x: x*2).collect())
    outPutPrint("rdd filter is :" , rdd.filter(lambda x : x > 50).collect())
    outPutPrint("flatmap ",rdd.map(lambda x:[x , x*10]).collect() , rdd.flatMap(lambda x:[x,x*10]).collect())
    outPutPrint("The numbers of rdd is " , rdd.count())
    distFile = sc.textFile('./data/README.txt')
    outPutPrint("the line of readme is ",distFile.count())
    outPutPrint("The sum of rdd are ",rdd.reduce(lambda x,y: x+y))
    outPutPrint("The ranked rdd elem are ",rdd.takeOrdered(3,lambda x:-1*x))
    
    kvData = [(1,3),('a',2),(3,'e'),(1,4)]
    kvrdd = sc.parallelize(kvData)
    outPutPrint("kvrdd key",kvrdd.keys().collect())
    outPutPrint("reduce by key add:" , kvrdd.reduceByKey(lambda x,y : x+y).collect())
    outPutPrint("sort by key :" , kvrdd.sortByKey().collect())
    outPutPrint("group by key : " ,kvrdd.groupByKey().collect())
    
    broadcastVar = sc.broadcast([1,4,6,7,4,3])
    outPutPrint("broadcastVar value is" , broadcastVar.value)
    
    accum = sc.accumulator(0)
    def f(x):
        global accum
        accum += x
    rdd.foreach(f)
    outPutPrint("accumlate " , accum.value)
    
    blankAccum = sc.accumulator(0)
    def countBlank(line):
        global blankAccum
        if line == "":
            blankAccum += 1
        return line.split(' ')
    textFileRDD = distFile.flatMap(countBlank)
    outPutPrint("blank lines :" , blankAccum.value)
    outPutPrint("text file is " , textFileRDD.take(10))
    
    
    #building a word count application
    wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
    wordsRDD = sc.parallelize(wordsList , 4)
    outPutPrint("type of wordsRDD :" , type(wordsRDD))
    def makePlural(word):
        return word + 's'
    pluralRDD = wordsRDD.map(makePlural)
    outPutPrint("Plural word results are :" ,pluralRDD.collect())
    outPutPrint("lambda function get the same result : " , wordsRDD.map(lambda x : x + 's').collect())
    outPutPrint("Each word lengths : ", wordsRDD.map(lambda x : len(x)).collect())
    #pair RDDs
    wordPairs = wordsRDD.map(lambda x : (x,1))
    outPutPrint("wordPairs are :" ,wordPairs.collect())
    #groupByKey
    wordsGrouped = wordPairs.groupByKey()
    outPutPrint("grouped wordPairs are :" , wordsGrouped.mapValues(list).collect())
    wordsCountGrouped = wordsGrouped.map(lambda (x,y):(x , sum(y)))
    outPutPrint("word Count :" , wordsCountGrouped.collect())
    
    #reduceBykey
    wordCounts = wordPairs.reduceByKey(lambda x, y : x + y)
    outPutPrint("word count reducebykey :" ,  wordCounts.collect())
    
    outPutPrint("union all paths :" ,wordsRDD.map(lambda x:(x,1)).reduceByKey(lambda x,y : x+y).collect())
    
    #count unique words 
    outPutPrint("unique words num are :" , wordsRDD.distinct().map(lambda x : 1).reduce(lambda x,y : x + y))
    
    def wordCount(wordListRDD):
        return wordListRDD.map(lambda x : (x , 1)).reduceByKey(lambda x,y : x + y)
    import re
    def removePunctuation(text):
        return re.sub(r'[^A-Za-z\s\d]' , '' , text).strip().lower()
    shakespeareRDD = sc.textFile('./data/Shakespeare.txt').map(removePunctuation)
    
    print '\n'.join(shakespeareRDD
                    .zipWithIndex()
                    .map(lambda (l , m) : '%d : %s' % (m ,l))
                    .take(10))
    shakespeareWordsRDD = shakespeareRDD.flatMap(lambda line : line.split(' '))
    shakespeareWordCount = shakespeareWordsRDD.count()
    outPutPrint('top 5 shakespeareWordsRDD elem' , shakespeareWordsRDD.top(5))
    outPutPrint('shakespeareWordCount ',shakespeareWordCount)
    shakeWordsRDD = shakespeareWordsRDD.filter(lambda x : x!= '')
    top15WordsAndCounts = (shakeWordsRDD
                           .map(lambda x:(x,1))
                           .reduceByKey(lambda x, y : x + y)
                           .takeOrdered(15 , lambda (x,y): -1*y))
    print '\n'.join(map( lambda (w ,c ):'%s : %d'%(w,c), top15WordsAndCounts))
    
    
except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()