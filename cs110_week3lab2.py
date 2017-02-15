#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:35:20 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime
import re
import seaborn

sc = SparkContext("local[4]" , "cs110 week3 lab2 APP")
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
    #loadData
    google = (sqc.read
              .csv('./data/targets.csv' , header = True)
              .rdd
              .map(lambda x : (x.asDict()['id'],'%s %s %s' % (x.asDict()['title'] , x.asDict()['description'] ,x.asDict()['manufacturer'])))
              )
    print google.take(2)
    google.cache()
    googleSmall = sc.parallelize(google.takeSample(False , 200 , 11))
    outPutPrint("google has items of ", google.count())
    
    amazon = (sqc.read
              .csv('./data/targets.csv' , header = True)
              .rdd
              .map(lambda x : (x.asDict()['id'],'%s %s %s' % (x.asDict()['title'] , x.asDict()['description'] ,x.asDict()['manufacturer'])))
              )
    print amazon.take(2)
    amazon.cache()
    print amazon.count()
    amazonSmall = sc.parallelize(amazon.takeSample(False , 200 , 11))
    
    split_regex = r'\W+'
    def simpleTokenize(string):
        if string == '':
            return []
        return re.split(split_regex,re.sub(split_regex , ' ' , string).strip().lower())
    
    def getStopWords(string):
        return re.split(r"u'(\W+)'" , string)
    
    stopwords = set(sc.textFile('./data/stopwords.txt').flatMap(getStopWords).collect())
    #print stopwords
    
    def tokenize(string):
        return [w for w in simpleTokenize(string) if w not in stopwords]
    
    #('b00004tkvy', ['noah', 'ark', 'activity', 'center', 'jewel', 'case', 'ages', '3', '8', 'victory', 'multimedia'])
    amazonRecToToken = amazonSmall.map(lambda (x,y):(x,tokenize(y)))
    googleRecToToken = googleSmall.map(lambda (x,y):(x,tokenize(y)))
    amazonToken = amazon.map(lambda (x,y):(x , tokenize(y)))
    googleToken = google.map(lambda (x,y):(x,tokenize(y)))
    def countTokens(venderRDD):
        return venderRDD.map(lambda (x,y):len(y)).reduce(lambda x,y:x+y)
    
    def findBiggestRecord(venderRDD):
        return venderRDD.sortBy(lambda (x,y):len(y)*-1)
    
    #TF-IDF
    def tf(tokens):
        tfResult = {}
        for t in tokens:
            tfResult.setdefault(t,0)
            tfResult[t] += 1
        tokens_num = len(tokens)
        for token in tfResult:
            tfResult[token] = tfResult[token]*1.0/tokens_num
                    
        return tfResult
    
    corpusRDD = amazonRecToToken.union(googleRecToToken)
    
    def idfs(corpus):
        uniqueTokens = corpus.map(lambda (x,y):list(set(y)))
        tokenCountPairTuple = uniqueTokens.flatMap(lambda x : x)
        tokenSumPairTuple = tokenCountPairTuple.map(lambda x:(x,1)).reduceByKey(lambda x,y :x+y)
        N = corpus.count()
        return (tokenSumPairTuple.map(lambda (x , y) :(x , N*1.0/y)))
    
    idfgoogleamazon = idfs(googleToken.union(amazonToken))
    outPutPrint("google and amazon all together has words of :" , idfgoogleamazon.count())
    print idfgoogleamazon.take(10)
    idfsWeight = idfgoogleamazon.collectAsMap()
    #draw idf
    seaborn.distplot(idfgoogleamazon.map(lambda (x,y) : y).collect())
    
    #tfidf
    def tfidf(tokens ,idf):
        '''
        tokens are word list
        idf is a dict
        '''
        tfs = tf(tokens)
        tfIdfDict = {}
        for t in tfs:
            tfIdfDict[t] = tfs[t] * idf[t]
        return tfIdfDict
    
    #tf is a dict ; idf is a RDD ; func tokenize return a list of word
    import math
    def dotprod(a,b):
        '''
        a,b is a dict
        calac dot product
        '''
        sumResult = 0
        for key in set(a.keys()).intersection(set(b.keys())):
          sumResult += a[key] * b[key]
        return sumResult
    def norm(a):
        '''
        square root of dot product
        '''
        return math.sqrt(dotprod(a,a))
    
    def cossim(a,b):
        '''
        a,b is a dict
        calc cos similarity
        '''
        return dotprod(a,b) / (norm(a)*norm(b))
    
    def cosineSimilarity(string1 , string2 , idfsDictionary):
        w1 = tfidf(tf(tokenize(string1)) , idfsDictionary)
        w2 = tfidf(tf(tokenize(string2)) , idfsDictionary)
        return cossim(w1, w2)
    
    #create descarte
    crossRDD = googleToken.cartesian(amazonToken).cache()
    
    def computeSimilarity(record):
        '''
        define a worker func
        '''
        googleRec = record[0]
        amazonRec = record[1]
        googleUrl = googleRec[0]
        amazonID = amazonRec[0]
        googleValue = googleRec[1]
        amazonValue = amazonRec[1]
        cs = cosineSimilarity(googleValue , amazonValue , idfsWeight)
        return (googleUrl , amazonID , cs)
    
    def computeSimilarityBroadcast(record):
        '''
        define a worker func
        '''
        googleRec = record[0]
        amazonRec = record[1]
        googleUrl = googleRec[0]
        amazonID = amazonRec[0]
        googleValue = googleRec[1]
        amazonValue = amazonRec[1]
        cs = cosineSimilarity(googleValue , amazonValue , idfsWeightBroadcast.value)
        return (googleUrl , amazonID , cs)
    
    idfsWeightBroadcast = sc.broadcast(idfsWeight)
    
    #evaluation
    goldStandard = sqc.read.csv('./data/mapping.csv' , header = True).rdd.map(lambda (x,y) : ('%s %s'% (x , y) , 'gold'))
    outPutPrint("goldStandard sample : " , goldStandard.take(2))
    
    amazonFullRecToToken = amazon.map(lambda (x,y):(x,tokenize(y)))
    googleFullRecToToken = google.map(lambda (x,y):(x,tokenize(y)))
    
    
except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()