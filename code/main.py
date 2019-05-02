# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:18:59 2019

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
import sklearn
import pickle

from functions import *
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor      
from datetime import datetime
from time import sleep

import csv
import random

app_name        = "WIFI_TX"   # WIFI_TX / WIFI_RX
sched_name      = "ETF"
classifier_type = "RT"
learning_type   = "IL"        # IL / SL

# First, set phase to "training" and run main.py
# It will generate models in the current working directory

# Then set the phase to "testing" and the script will use the generated models 
# to test the training accuracy of the model
# It will generate CSV files containing training accuracies

#phase = "testing"
phase = "training"

max_tree_depth_array = [4, 8, 16, 32, 48, 64, 128]

env = ApplicationEnv(app_name, sched_name)

datafile_name = app_name + "_data.csv"

alldata = pd.read_csv("../Datasets/" + datafile_name, header = 0)


if learning_type == "IL":
    feature_data = alldata.loc[:, env.IL_feature_names]
elif learning_type == "SL":
    feature_data = alldata.loc[:, env.SL_feature_names]
else:
    raise Exception('Unsupported learning type')

labels = alldata.loc[:, "resource"]

for max_tree_depth in max_tree_depth_array:

    filename = classifier_type + "_model_" + app_name + "_" + str(max_tree_depth) + "_" + learning_type + ".sav"
    
    
    if phase == "training":
        model = env.f_train_model(feature_data,  labels, classifier_type, max_tree_depth)
        pickle.dump(model, open(filename, 'wb'))
        
    elif phase == "testing":
        regressor = pickle.load(open(filename, 'rb'))
        output_labels = env.f_test_model(feature_data, regressor)
        output_labels_modified = np.floor(output_labels)
        
        #calculate accuracy
        #compare between output_labels and labels
        correct_pred = (labels == output_labels_modified)
        pe_accuracy = 100*sum(correct_pred)/len(correct_pred)
        
        #calculate cluster accuracy
        #compare original cluster and predicted clusters
        orig_cluster = env.f_cluster_output(labels)
        predicted_cluster = env.f_cluster_output(output_labels_modified)
        
        correct_clusters = (orig_cluster == predicted_cluster)
        cluster_accuracy = 100*sum(correct_clusters)/len(correct_clusters)
        
        print ('PE accuracy: ', pe_accuracy, '; Cluster accuracy: ', cluster_accuracy)
        
        accuracy = np.zeros(shape=(1,2))
        accuracy[0, 0] = pe_accuracy
        accuracy[0, 1] = cluster_accuracy
        
        np.savetxt(app_name + '_' + classifier_type + '_' + str(max_tree_depth) + "_" + learning_type + "_accuracy.csv", accuracy, delimiter=",", header = "PE, Cluster", comments='')
                
    else:
        raise Exception('Unexpected phase. It should be either training or testing')
