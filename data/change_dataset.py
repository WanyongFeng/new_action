# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:58:35 2021

@author: wanyong
"""
import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import pickle 
import tensorflow as tf
import numpy as np
from models import get_model
from utils.hparams import HParams

dfile = 'C:/Users/wanyong/Desktop/gsmrl_imputation/data/cube5.pkl'
split = 'train'
with open(dfile, 'rb') as f:
            data_dict = pickle.load(f)
data, label = data_dict[split]
b = np.zeros((data.shape[0],data.shape[1]))
for i, vals in enumerate(data):
    for j, val in enumerate(vals):
        if not val == 0:
            b[i][j] = 1

data = data[:2, :]
b = b[:2, :]          
g = tf.Graph()
with g.as_default():
    # open a session
    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=g)
    # build ACFlow model
    model_hps = HParams('C:/Users/wanyong/Desktop/gsmrl_wanyong/exp/gas/vec/params.json')
    model = get_model(sess, model_hps)
    # restore weights
    saver = tf.train.Saver()
    restore_from = 'C:/Users/wanyong/Desktop/gsmrl_wanyong/exp/gas/vec//weights/params.ckpt'
    saver.restore(sess, restore_from)
    sam = model.run(
    [model.sam],
        feed_dict={model.x: data,
        model.b: b,
        model.m: np.ones_like(b)})  
    print(data)
    print('########')
    print(np.mean(sam[0][0],axis = 0))
    print(np.mean(sam[0][1],axis = 0))
# =============================================================================
#     print(np.mean(sam[0][0], axis = 0))
# =============================================================================
