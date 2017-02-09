#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:28:58 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime
from pyspark.sql.functions import udf ,split , regexp_extract ,col , sum , lit ,concat


sc = SparkContext("local[3]","cs105 Lecture App")
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


outPutPrint("[INFO] the spark cluster started")

try:
    dataFile = "./data/access_log_Jul95"
    data = sqc.read.text(dataFile)
    outPutPrint("show first log" , data.first())
    split_df = data.select(regexp_extract('value',r'^([^\s]+\s)' ,1).alias('host'),\
                           regexp_extract('value',r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]' ,1).alias('timestamp'),\
                           regexp_extract('value', r'^.*"\w+\s+([^\s]+)\s+HTTP.*"', 1).alias('path'),\
                           regexp_extract('value', r'^.*"\s+([^\s]+)',1).cast('integer').alias('status'),\
                           regexp_extract('value', r'^.*\s+(\d+)$',1).cast('integer').alias('content_size') )
    split_df.show()
    outPutPrint("count all null data is:", data.filter(data.value.isNull()).count())
    bad_rows_df = split_df.filter(split_df['host'].isNull() | \
                              split_df['timestamp'].isNull() | \
                              split_df['path'].isNull() | \
                              split_df['status'].isNull() | \
                              split_df['content_size'].isNull()
                              )
    outPutPrint("count bad row is : " , bad_rows_df.count())
    def count_null(col_name):
        return sum(col(col_name).isNull().cast('integer')).alias(col_name)
    exps = []
    for c in split_df.columns:
        exps.append(count_null(c))
        
    split_df.agg(*exps).show()
    
    bad_content_size_df = data.filter( ~data['value'].rlike(r'\d+$'))
    outPutPrint("invalid data count is :" ,bad_content_size_df.count())
    bad_content_size_df.select(concat(bad_content_size_df['value'] , lit("*"))).show(truncate = False)
    
    cleaned_df = split_df.na.fill({'content_size':0 , 'status':0})
    exp_c = []
    for c in cleaned_df.columns:
        exp_c.append(count_null(c))
    cleaned_df.agg(*exp_c).show()
    
    month_map = {
              'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
              'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
              }
    
    def parse_clf_time(s):
        return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
                                                                      int(s[7:11]),
                                                                      month_map[s[3:6]],
                                                                      int(s[0:2]),
                                                                      int(s[12:14]),
                                                                      int(s[15:17]),
                                                                      int(s[18:20])
                                                                    )
        
    u_parse_time = udf(parse_clf_time)
    logs_df = cleaned_df.select('*' , u_parse_time(cleaned_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
    total_log_entries = logs_df.count()
    logs_df.show()
    outPutPrint("cleaned data count is " , total_log_entries)
    logs_df.printSchema()
    
except Exception as e:
    outPutPrint("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")
    clear_env()