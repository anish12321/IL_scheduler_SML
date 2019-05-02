# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:19:36 2019

@author: skmandal, anallam1
"""

import os
import math
import shutil
import random
import scipy.io
import numpy as np
import pandas as pd                
import matplotlib.pyplot as plt
#import sklearn

from functions import *
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression   
from datetime import datetime
from time import sleep

import csv
import random

#for plotting
from IPython.display import clear_output 
# for Earlystop

class ApplicationEnv:
    def __init__(self, chooseApp, chooseSched):

        # Get application and scheduler configuration:
        self.chooseApp = chooseApp
        self. chooseSched = chooseSched
        
        # Total of columns: 9
        self.col_name_lst = ["timestamps", "min_exec_time", "max_exec_time", "mean_exec_time", "median_exec_time",
                             "var_exec_time", "downward_depth", "relative_job_id", 
                             "PE_0_free", "PE_1_free", "PE_2_free", "PE_3_free", "PE_4_free", "PE_5_free", "PE_6_free",
                             "PE_7_free", "PE_8_free", "PE_9_free", "PE_10_free", "PE_11_free", "PE_12_free", "PE_13_free",
                             "PE_14_free", "PE_15_free", "PE_16_free", 
                             "min_comm_time_PE_0", "max_comm_time_PE_0", "avg_comm_time_PE_0", "med_comm_time_PE_0", "var_comm_time_PE_0",
                             "min_comm_time_PE_1", "max_comm_time_PE_1", "avg_comm_time_PE_1", "med_comm_time_PE_1", "var_comm_time_PE_1",
                             "min_comm_time_PE_2", "max_comm_time_PE_2", "avg_comm_time_PE_2", "med_comm_time_PE_2", "var_comm_time_PE_2",
                             "min_comm_time_PE_3", "max_comm_time_PE_3", "avg_comm_time_PE_3", "med_comm_time_PE_3", "var_comm_time_PE_3",
                             "min_comm_time_PE_4", "max_comm_time_PE_4", "avg_comm_time_PE_4", "med_comm_time_PE_4", "var_comm_time_PE_4",
                             "min_comm_time_PE_5", "max_comm_time_PE_5", "avg_comm_time_PE_5", "med_comm_time_PE_5", "var_comm_time_PE_5",
                             "min_comm_time_PE_6", "max_comm_time_PE_6", "avg_comm_time_PE_6", "med_comm_time_PE_6", "var_comm_time_PE_6",
                             "min_comm_time_PE_7", "max_comm_time_PE_7", "avg_comm_time_PE_7", "med_comm_time_PE_7", "var_comm_time_PE_7",
                             "min_comm_time_PE_8", "max_comm_time_PE_8", "avg_comm_time_PE_8", "med_comm_time_PE_8", "var_comm_time_PE_8",
                             "min_comm_time_PE_9", "max_comm_time_PE_9", "avg_comm_time_PE_9", "med_comm_time_PE_9", "var_comm_time_PE_9",
                             "min_comm_time_PE_10", "max_comm_time_PE_10", "avg_comm_time_PE_10", "med_comm_time_PE_10", "var_comm_time_PE_10",
                             "min_comm_time_PE_11", "max_comm_time_PE_11", "avg_comm_time_PE_11", "med_comm_time_PE_11", "var_comm_time_PE_11",
                             "min_comm_time_PE_12", "max_comm_time_PE_12", "avg_comm_time_PE_12", "med_comm_time_PE_12", "var_comm_time_PE_12",
                             "min_comm_time_PE_13", "max_comm_time_PE_13", "avg_comm_time_PE_13", "med_comm_time_PE_13", "var_comm_time_PE_13",
                             "min_comm_time_PE_14", "max_comm_time_PE_14", "avg_comm_time_PE_14", "med_comm_time_PE_14", "var_comm_time_PE_14",
                             "min_comm_time_PE_15", "max_comm_time_PE_15", "avg_comm_time_PE_15", "med_comm_time_PE_15", "var_comm_time_PE_15",
                             "min_comm_time_PE_16", "max_comm_time_PE_16", "avg_comm_time_PE_16", "med_comm_time_PE_16", "var_comm_time_PE_16",
                             "resource"]
        
        self.col = {k: v for v, k in enumerate(self.col_name_lst)}
       
        # Set of features to be selected
        self.IL_feature_names = ["min_exec_time", "max_exec_time", "mean_exec_time", "median_exec_time",
                             "var_exec_time", "downward_depth", "relative_job_id",
                             "PE_0_free", "PE_1_free", "PE_2_free", "PE_3_free", "PE_4_free", "PE_5_free", "PE_6_free",
                             "PE_7_free", "PE_8_free", "PE_9_free", "PE_10_free", "PE_11_free", "PE_12_free", "PE_13_free",
                             "PE_14_free", "PE_15_free", "PE_16_free", 
                             "min_comm_time_PE_0", "max_comm_time_PE_0", "avg_comm_time_PE_0", "med_comm_time_PE_0", "var_comm_time_PE_0",
                             "min_comm_time_PE_1", "max_comm_time_PE_1", "avg_comm_time_PE_1", "med_comm_time_PE_1", "var_comm_time_PE_1",
                             "min_comm_time_PE_2", "max_comm_time_PE_2", "avg_comm_time_PE_2", "med_comm_time_PE_2", "var_comm_time_PE_2",
                             "min_comm_time_PE_3", "max_comm_time_PE_3", "avg_comm_time_PE_3", "med_comm_time_PE_3", "var_comm_time_PE_3",
                             "min_comm_time_PE_4", "max_comm_time_PE_4", "avg_comm_time_PE_4", "med_comm_time_PE_4", "var_comm_time_PE_4",
                             "min_comm_time_PE_5", "max_comm_time_PE_5", "avg_comm_time_PE_5", "med_comm_time_PE_5", "var_comm_time_PE_5",
                             "min_comm_time_PE_6", "max_comm_time_PE_6", "avg_comm_time_PE_6", "med_comm_time_PE_6", "var_comm_time_PE_6",
                             "min_comm_time_PE_7", "max_comm_time_PE_7", "avg_comm_time_PE_7", "med_comm_time_PE_7", "var_comm_time_PE_7",
                             "min_comm_time_PE_8", "max_comm_time_PE_8", "avg_comm_time_PE_8", "med_comm_time_PE_8", "var_comm_time_PE_8",
                             "min_comm_time_PE_9", "max_comm_time_PE_9", "avg_comm_time_PE_9", "med_comm_time_PE_9", "var_comm_time_PE_9",
                             "min_comm_time_PE_10", "max_comm_time_PE_10", "avg_comm_time_PE_10", "med_comm_time_PE_10", "var_comm_time_PE_10",
                             "min_comm_time_PE_11", "max_comm_time_PE_11", "avg_comm_time_PE_11", "med_comm_time_PE_11", "var_comm_time_PE_11",
                             "min_comm_time_PE_12", "max_comm_time_PE_12", "avg_comm_time_PE_12", "med_comm_time_PE_12", "var_comm_time_PE_12",
                             "min_comm_time_PE_13", "max_comm_time_PE_13", "avg_comm_time_PE_13", "med_comm_time_PE_13", "var_comm_time_PE_13",
                             "min_comm_time_PE_14", "max_comm_time_PE_14", "avg_comm_time_PE_14", "med_comm_time_PE_14", "var_comm_time_PE_14",
                             "min_comm_time_PE_15", "max_comm_time_PE_15", "avg_comm_time_PE_15", "med_comm_time_PE_15", "var_comm_time_PE_15",
                             "min_comm_time_PE_16", "max_comm_time_PE_16", "avg_comm_time_PE_16", "med_comm_time_PE_16", "var_comm_time_PE_16"]

        self.IL_f_name = {k: v for v, k in enumerate(self.IL_feature_names)}
        
        self.SL_feature_names = ["min_exec_time", "max_exec_time", "mean_exec_time", "median_exec_time",
                             "var_exec_time", "downward_depth", 
                             "min_comm_time_PE_0", "max_comm_time_PE_0", "avg_comm_time_PE_0", "med_comm_time_PE_0", "var_comm_time_PE_0",
                             "min_comm_time_PE_1", "max_comm_time_PE_1", "avg_comm_time_PE_1", "med_comm_time_PE_1", "var_comm_time_PE_1",
                             "min_comm_time_PE_2", "max_comm_time_PE_2", "avg_comm_time_PE_2", "med_comm_time_PE_2", "var_comm_time_PE_2",
                             "min_comm_time_PE_3", "max_comm_time_PE_3", "avg_comm_time_PE_3", "med_comm_time_PE_3", "var_comm_time_PE_3",
                             "min_comm_time_PE_4", "max_comm_time_PE_4", "avg_comm_time_PE_4", "med_comm_time_PE_4", "var_comm_time_PE_4",
                             "min_comm_time_PE_5", "max_comm_time_PE_5", "avg_comm_time_PE_5", "med_comm_time_PE_5", "var_comm_time_PE_5",
                             "min_comm_time_PE_6", "max_comm_time_PE_6", "avg_comm_time_PE_6", "med_comm_time_PE_6", "var_comm_time_PE_6",
                             "min_comm_time_PE_7", "max_comm_time_PE_7", "avg_comm_time_PE_7", "med_comm_time_PE_7", "var_comm_time_PE_7",
                             "min_comm_time_PE_8", "max_comm_time_PE_8", "avg_comm_time_PE_8", "med_comm_time_PE_8", "var_comm_time_PE_8",
                             "min_comm_time_PE_9", "max_comm_time_PE_9", "avg_comm_time_PE_9", "med_comm_time_PE_9", "var_comm_time_PE_9",
                             "min_comm_time_PE_10", "max_comm_time_PE_10", "avg_comm_time_PE_10", "med_comm_time_PE_10", "var_comm_time_PE_10",
                             "min_comm_time_PE_11", "max_comm_time_PE_11", "avg_comm_time_PE_11", "med_comm_time_PE_11", "var_comm_time_PE_11",
                             "min_comm_time_PE_12", "max_comm_time_PE_12", "avg_comm_time_PE_12", "med_comm_time_PE_12", "var_comm_time_PE_12",
                             "min_comm_time_PE_13", "max_comm_time_PE_13", "avg_comm_time_PE_13", "med_comm_time_PE_13", "var_comm_time_PE_13",
                             "min_comm_time_PE_14", "max_comm_time_PE_14", "avg_comm_time_PE_14", "med_comm_time_PE_14", "var_comm_time_PE_14",
                             "min_comm_time_PE_15", "max_comm_time_PE_15", "avg_comm_time_PE_15", "med_comm_time_PE_15", "var_comm_time_PE_15",
                             "min_comm_time_PE_16", "max_comm_time_PE_16", "avg_comm_time_PE_16", "med_comm_time_PE_16", "var_comm_time_PE_16"]
        
        self.SL_f_name = {k: v for v, k in enumerate(self.SL_feature_names)}
        

    def f_train_model(self, feature_data,  labels, classifier_type, max_tree_depth):
        
        X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.20, train_size=0.80, random_state=0)
        
        if classifier_type == 'RT':
            regressor = DecisionTreeRegressor(max_depth = max_tree_depth)
            regressor.fit(X_train, y_train)
        elif classifier_type == 'LR':
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
        else:
            raise Exception('Unexpected classifier type')
            
        return regressor
    
    def f_test_model(self, feature_data, regressor):
        
        output_labels = regressor.predict(feature_data)
        
        return output_labels
    
    def f_cluster_output(self, labels):
        
        # PE to Cluster Mapping
        # 0-3   -> 1
        # 4-7   -> 2
        # 8-9   -> 3
        # 10-13 -> 4
        # 14-15 -> 5
        
        num_labels = len(labels)
        
        cluster_labels = np.zeros(shape=(num_labels,1))
        
        for label_idx in range (0, num_labels):
            if labels[label_idx] == 0 or labels[label_idx] == 1 or labels[label_idx] == 2 or labels[label_idx] == 3 :
                cluster_labels[label_idx] = 1
            elif labels[label_idx] == 4 or labels[label_idx] == 5 or labels[label_idx] == 6 or labels[label_idx] == 7 :
                cluster_labels[label_idx] = 2
            elif labels[label_idx] == 8 or labels[label_idx] == 9 :
                cluster_labels[label_idx] = 3
            elif labels[label_idx] == 10 or labels[label_idx] == 11 or labels[label_idx] == 12 or labels[label_idx] == 13 :
                cluster_labels[label_idx] = 4
            elif labels[label_idx] == 14 or labels[label_idx] == 15 :
                cluster_labels[label_idx] = 5
            elif labels[label_idx] == 16 :
                cluster_labels[label_idx] = 6
            else:
                raise Exception ('Unexpected label')
                
        return cluster_labels
                    
