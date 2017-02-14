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
    pass
except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()