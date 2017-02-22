#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:52:29 2017

@author: hp
"""

f = open('./data/train.txt','r')
r = open('./data/ctr.txt','w')
for i in xrange(1000000):
    s = f.readline()
    print s
    r.write(s)
    
f.close()
r.close()