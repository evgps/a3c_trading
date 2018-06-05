# -*- coding: utf-8 -*-
CUDA_VISIBLE_DEVICES=-1 
import os
import pandas as pd
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
import pybacktest as pb
import matplotlib.pyplot as plt
import threading
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
import sys
from trader_gym import environment
from A3C_class import *
from configs import postfix, load_model, LR, dep

#Если нет файла, то его нужно создать с помощью load_data.ipynb
R = pd.read_pickle('data/'+str(postfix)+'w5k') #уже нормировванные
old_R = R.copy()
print(R.values.shape)
#R = R[:10000]
vals = R.values
D = np.hstack([vals[i:-dep+i-1, :] for i in range(dep,0,-1)])
RR = pd.DataFrame(D, R[dep:-1].index)
LENGTH = RR.shape[0]
#R = RR[:LENGTH]
# Configs
# Main
max_episode_length = LENGTH
tf.reset_default_graph()
if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device('/cpu:0'): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate = LR, decay = 0.99, epsilon = 1e-6)
    master_network = AC_Network(s_size,a_size,'global',None)
    num_workers = 10
    workers = []
    for i in range(num_workers):
        env = environment(RR, LENGTH)
        workers.append(Worker(env, i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=25)
    
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()     
if 'sess' in locals() and sess is not None:
    print('Close interactive session')
    sess.close()    

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)) as sess:   
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("tb/train", graph=sess.graph)
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver, dep)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)