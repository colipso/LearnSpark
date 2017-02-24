#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:27:15 2017

@author: hp
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import traceback
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn


sc = SparkContext("local[4]","cs120 week4")
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
    def create_2D_gaussian(mn, variance, cov, n):
        """Randomly sample points from a two-dimensional Gaussian distribution"""
        return np.random.multivariate_normal([mn,mn],np.array([[variance , cov],[cov,variance]]),n)
    Log("test create 2D gaussian:" , create_2D_gaussian(5,23,3,10) )
    data_random = create_2D_gaussian(50 , 1 , 0 , 100)
    random_x = data_random[:,0]
    random_y = data_random[:,1]
    grid = seaborn.JointGrid(random_x, random_y, space=0, size=6, ratio=50)
    grid.plot_joint(plt.scatter, color="g")
    grid.plot_marginals(seaborn.rugplot, height=1, color="g")
    
    data_correlate = create_2D_gaussian(50 , 1 , 0.8 , 100)
    cor_x = data_correlate[:,0]
    cor_y = data_correlate[:,1]
    grid = seaborn.JointGrid(cor_x, cor_y, space=0, size=6, ratio=50)
    grid.plot_joint(plt.scatter, color="g")
    grid.plot_marginals(seaborn.rugplot, height=1, color="g")
    
    #center data
    correlated_data = sc.parallelize(data_correlate)
    mean_correlated = correlated_data.mean()
    correlated_data_zero_mean = correlated_data.map(lambda x : x - mean_correlated)
    Log("mean of correlated data :" , mean_correlated)
    
    correlated_cov = correlated_data_zero_mean.map(lambda x:np.outer(x,x)).reduce(lambda x, y : x+y)/correlated_data_zero_mean.count()
    Log("correlated cov are ",correlated_cov)
    def estimate_covariance(data):
        """Compute the covariance matrix for a given rdd.
    
        Note:
            The multi-dimensional covariance array should be calculated using outer products.  Don't
            forget to normalize the data by first subtracting the mean.
    
        Args:
            data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.
    
        Returns:
            np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
                length of the arrays in the input `RDD`.
        """
        cols_mean = data.mean()
        return data.map(lambda x : x-cols_mean).map(lambda x : np.outer(x,x)).reduce(lambda x, y : x+y)/data.count()
    
    correlated_cov_auto = estimate_covariance(correlated_data)
    Log("use func to calc correlated cov are : " , correlated_cov_auto)
    eig_vals, eig_vecs = np.linalg.eigh(correlated_cov_auto)
    Log("eig vals is " , eig_vals)
    Log("eig vecs are " , eig_vecs)
    
    def pca(data, k=2):
        """Computes the top `k` principal components, corresponding scores, and all eigenvalues.
    
        Note:
            All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
            each eigenvectors as a column.  This function should also return eigenvectors as columns.
    
        Args:
            data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
            k (int): The number of principal components to return.
    
        Returns:
            tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
                scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
                rows equals the length of the arrays in the input `RDD` and the number of columns equals
                `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
                of length `k`.  Eigenvalues is an array of length d (the number of features).
        """
        cols_mean = data.mean()
        corviance = data.map(lambda x : x - cols_mean).map(lambda x : np.outer(x , x)).reduce(lambda x , y : x + y) /  data.count()
        eig_vals, eig_vecs = np.linalg.eigh(corviance,UPLO='U')
        indis_r = np.argsort(eig_vals)
        indis = indis_r[::-1][:k]
        '''
        # Return the `k` principal components, `k` scores, and all eigenvalues
        #print eig_vals
        #print eig_vecs
        #print eig_vals[::-1]
        n,z = np.shape(eig_vecs)
        result_eig_vecs = np.zeros(np.shape(eig_vecs))
        for i in range(n):
          for j in range(z):
            result_eig_vecs[i,j] = eig_vecs[n-1-i,z-1-j]
        print eig_vecs
        print result_eig_vecs
        print result_eig_vecs
        #print eig_vecs[::-1]
        
        #print indis_r
        #print eig_vecs
        #print eig_vecs[indis]
        '''
        k_principal = eig_vecs[:,indis]
        k_scores = data.map(lambda x : np.dot(x,k_principal))
        return (k_principal , k_scores , eig_vals[::-1])
    
    #use PCA
    random_data_rdd = sc.parallelize(data_random)
    top_components_random, random_data_scores_auto, eigenvalues_random = pca(random_data_rdd,2)
    
    Log("top 2 components of random data are " , top_components_random)
    Log("3 examples of transformed random data are " , random_data_scores_auto.take(3))
    Log("eigh values are " , eigenvalues_random)
    
    #variance explained
    def variance_explained(data, k=1):
        """Calculate the fraction of variance explained by the top `k` eigenvectors.
    
        Args:
            data (RDD of np.ndarray): An RDD that contains NumPy arrays which store the
                features for an observation.
            k: The number of principal components to consider.
    
        Returns:
            float: A number between 0 and 1 representing the percentage of variance explained
                by the top `k` eigenvectors.
        """
        components, scores, eigenvalues = pca(data,k)
        return sum(eigenvalues[:k]) *1.0 / sum(eigenvalues)

    Log("Percentage of variance explained by the first component of random data is %.2f " % variance_explained(random_data_rdd ,1))
    
    
except Exception as e:
    Log("[ERROR]",e)
    traceback.print_exc()
finally:
    #time.sleep(30)#for check sparkUI before cluster closed
    sc.stop()
    Log("[INFO]The spark cluster stopped")
    clear_env()