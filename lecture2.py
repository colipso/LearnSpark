#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:01:39 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import time

sc = SparkContext("local[2]","cs105Lecture App")
sc.setLogLevel("ERROR")
sqc = SQLContext(sc)

def outPutPrint(*context):
    outputlogo = "---->"
    string_print = ""
    for c in context:
        string_print += str(c)+"; "
    print outputlogo,string_print,'\n'
    return True

outPutPrint("[INFO] the spark cluster started")


try:
    pass


except Exception as e:
    outPutPrint("[ERROR]",e)
finally:
    time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    outPutPrint("[INFO]The spark cluster stopped")