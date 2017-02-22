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

sc = SparkContext("local[4]","cs120 week3")
sc.setLogLevel("ERROR")
sqc = SQLContext(sc)

def log(*context):
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
    log("Free Memory is ",memNotUsed_G , "G")
    return memNotUsed_G 


def clear_env():
    '''
    for prevent memory error,clear python veriable
    '''
    for key in globals().keys():
        if key not in ["memory_stat" , "log" , "time" , "datetime"]:
            if not key.startswith("__"):
                globals().pop(key)
    memory_stat()
    log("The python env is cleared")
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
    log(a)
    #load data
    raw_df = sqc.read.text('./data/test.txt').withColumnRenamed('value','text')
    raw_df.printSchema()
    raw_df.show()
    
    
    
except Exception as e:
    log("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    log("[INFO]The spark cluster stopped")
    clear_env()