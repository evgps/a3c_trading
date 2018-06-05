# -*- coding: utf-8 -*-
import os
import pandas as pd
from multiprocessing import Pool
import warnings
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
from configs import TRAIN_DATA, LOAD_MODEL, LR, FRAMES_STACKED, NUM_WORKERS, MODEL_DIR
warnings.filterwarnings("ignore")

# Если нет файла, то его нужно создать с помощью load_data.ipynb

train_df = pd.read_pickle(TRAIN_DATA)

if FRAMES_STACKED > 1:
    data = np.hstack([train_df.values[i:-FRAMES_STACKED + i - 1, :] for i in range(FRAMES_STACKED, 0, -1)])
else:
    data = train_df.values

train_df = pd.DataFrame(data, train_df[FRAMES_STACKED:-1].index)
max_episode_len = train_df.shape[0]

tf.reset_default_graph()
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with tf.device(USE_DEVICE):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=0.99, epsilon=1e-6)
    master_network = AC_Network(s_size, a_size, 'global', None)
    workers = []
    for i in range(NUM_WORKERS):
        env = environment(train_df, max_episode_len)
        workers.append(Worker(env, i, s_size, a_size, trainer, MODEL_DIR, global_episodes))
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
    if LOAD_MODEL:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("tb/train", graph=sess.graph)
    worker_threads = []
    for worker in workers:
        def worker_work(): return worker.work(max_episode_len, gamma, sess, coord, saver, FRAMES_STACKED)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
