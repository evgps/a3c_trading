# -*- coding: utf-8 -*-
CUDA_VISIBLE_DEVICES = -1
import os
import pandas as pd
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
import matplotlib as mpl
mpl.use('Agg')
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
# %load_ext ipycache
max_episode_length = 300
import matplotlib
matplotlib.rcParams.update({'font.size': 21})
from configs import postfix, test_postfix, load_model, LR, dep, test_postfix_real, postfix_real

data_dir_path = '/mnt/a3c_data/'

# data
print(data_dir_path + str(test_postfix))
R = pd.read_pickle(data_dir_path + str(test_postfix))
R = R[False == R.index.duplicated()]
test_R = pd.read_pickle(data_dir_path + str(test_postfix_real))
test_R = test_R[False == test_R.index.duplicated()]
# print (R.mean()[0], R.max()[0], (R.max()[0] - R.min()[0]), (R.max()[0] - R.mean()[0])/(R.max()[0] - R.min()[0]), (R.min()[0] - R.mean()[0])/(R.max()[0] - R.min()[0]))
R = (R - R.mean()) / (R.max() - R.min())
old_R = R.copy()
vals = R.values
D = np.hstack([vals[i:-dep + i - 1, :] for i in range(dep, 0, -1)])
R = pd.DataFrame(D, R[dep:-1].index)
# R = R[:10000]


# Main
def run_trades(max_episode_length, gamma, s_size, a_size, load_model, model_path, length, env):
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.RMSPropOptimizer(learning_rate=1e-3, decay=0.99, epsilon=1e-6)
        master_network = AC_Network(s_size, a_size, 'global', None)
        num_workers = 1
        workers = []
        for i in range(num_workers):
            workers.append(Test_Worker(env, i, s_size, a_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False, gpu_options=gpu_options)) as sess:
        coord = tf.train.Coordinator()

        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('tb/train', graph=sess.graph)
        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(max_episode_length, gamma, sess, coord, saver, dep, 0.33)
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)

        coord.join(worker_threads)
        acts = workers[0].acts
        rews = workers[0].rewards
    return acts, rews


# Торговая система - покупаем/продаем в зависимости от предсказанного класса
def backtest(acts, o_R):
    pacts = np.asarray(acts).copy()
    for i in range(1, pacts.shape[0]):
        if pacts[i] == 0:
            if pacts[i - 1] == -1:
                # from short
                pacts[i] = 0.5
            elif pacts[i - 1] == 1:
                pacts[i] = -0.5

    # plt.plot(pacts, '.')
    data = o_R[dep - 1:-3]

    buy = cover = pd.Series(pacts == 1, index=data.index)
    short = sell = pd.Series(pacts == -1, index=data.index)
    cover = pd.Series(pacts > 0, index=data.index)
    sell = pd.Series(pacts < 0, index=data.index)
    # print(np.sum(buy-sell))
    OHLC = data[[('DealPrice', 'close'), ('DealPrice', 'close'), ('DealPrice', 'close'), ('DealPrice', 'close')]]
    OHLC.columns = ['O', 'H', 'L', 'C']
    # Основной график эквити
    bt = pb.Backtest(locals())
    return bt


load_model = True
model_path = 'model'
length = R.shape[0]
noise_level = 1e-5
price = R.values[:, 3]
n_noises = 1
n_runs = 1

test_acts = np.zeros((n_noises, n_runs, length - 1))
test_rews = np.zeros((n_noises, n_runs, length - 1))

for j in range(n_noises):
    np.random.seed(j)
    env = environment(R + (np.random.rand(R.shape[0], R.shape[1]) - 0.5) * (1e-5 * (0**j)), length)
    for i in range(n_runs):
        np.random.seed(i + 1377)
        test_acts[j, i, :], test_rews[j, i, :] = run_trades(
            max_episode_length, gamma, s_size, a_size, load_model, model_path, length, env)
        print(j, i)

# ПРИ НЕУВЕРЕННОСТИ НУЖНО ДЕЛАТЬ ПРЕДЫДУЩЕЕ ДЕЙСТВИЕ, А НЕ 0

test_acts_probs = np.zeros((n_noises, a_size, length - 1))
tresh = 0.0
acts = np.zeros((n_noises, length - 1))
test_acts = test_acts.astype(int)
for k in range(n_noises):
    for j in range(n_runs):
        for i in range(length - 1):
            a = test_acts[k, j, i]
            test_acts_probs[k, a + 1, i] += 1
    test_acts_probs[k, :, :] = test_acts_probs[k, :, :] / n_runs
    acts[k, :] = [np.argmax(x) - 1 if max(x) > tresh else 0 for x in test_acts_probs[k, :, :].T]
# print(test_acts_probs)
# plt.plot(acts.T)
# plt.show()


def write_report(r, filename):
    with open(filename, "a") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line, file=input_file)


folder = os.path.relpath(".", "..")
for i in range(2):
    # [ R.index.get_loc('2015-12-29 10:06:00')[0], R.index.get_loc('2016-01-15 10:06:00')[0], \
    lengths = [R.index.get_loc('2016-03-15 10:06:00')[0], R.index.get_loc('2016-06-15 10:06:00')[0]]
    #names = ['2 weeks', '1 month', '3 month', '6 months']
    names = ['3 month', '6 months']
    bt = backtest(acts[-1, :lengths[i]], test_R[:lengths[i] + dep + 2])
    fig = plt.figure(figsize=(12, 10))
    plt.ylim(60000, 140000)
    bt.plot_trades()
    plt.legend(['long enter', 'short enter', 'long exit', 'short exit', 'equity', 'price'], loc='best')
    plt.title(folder + ' test on ' + names[i], fontname="Times New Roman")
    plt.ylabel('Rubles', fontname="Times New Roman")
    plt.xlabel('Time', fontname="Liberation Serif")
    plt.plot()
    # plt.show()
    plt.savefig('../new_plots/' + folder + '_' + names[i] + '.pdf', bbox_inches='tight', format='pdf')

    # Отчет по системе
    B = bt.report
    write_report(B, '../new_plots/' + folder + '_.txt')


# data
R = pd.read_pickle(data_dir_path + str(postfix))
R = R[False == R.index.duplicated()]

test_R = pd.read_pickle(data_dir_path + str(postfix_real))
test_R = test_R[False == test_R.index.duplicated()]
# print (R.mean()[0], R.max()[0], (R.max()[0] - R.min()[0]), (R.max()[0] - R.mean()[0])/(R.max()[0] - R.min()[0]), (R.min()[0] - R.mean()[0])/(R.max()[0] - R.min()[0]))
R = (R - R.mean()) / (R.max() - R.min())
old_R = R.copy()
vals = R.values
D = np.hstack([vals[i:-dep + i - 1, :] for i in range(dep, 0, -1)])
R = pd.DataFrame(D, R[dep:-1].index)
# R = R[:10000]


# Main
def run_trades(max_episode_length, gamma, s_size, a_size, load_model, model_path, length, env):
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.RMSPropOptimizer(learning_rate=1e-3, decay=0.99, epsilon=1e-6)
        master_network = AC_Network(s_size, a_size, 'global', None)
        num_workers = 1
        workers = []
        for i in range(num_workers):
            workers.append(Test_Worker(env, i, s_size, a_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False, gpu_options=gpu_options)) as sess:
        coord = tf.train.Coordinator()

        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('tb/train', graph=sess.graph)
        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(max_episode_length, gamma, sess, coord, saver, dep, 0.33)
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)

        coord.join(worker_threads)
        acts = workers[0].acts
        rews = workers[0].rewards
    return acts, rews


# Торговая система - покупаем/продаем в зависимости от предсказанного класса
def backtest(acts, o_R):
    pacts = np.asarray(acts).copy()
    for i in range(1, pacts.shape[0]):
        if pacts[i] == 0:
            if pacts[i - 1] == -1:
                # from short
                pacts[i] = 0.5
            elif pacts[i - 1] == 1:
                pacts[i] = -0.5

    # plt.plot(pacts, '.')
    data = o_R[dep - 1:-3]

    buy = cover = pd.Series(pacts == 1, index=data.index)
    short = sell = pd.Series(pacts == -1, index=data.index)
    cover = pd.Series(pacts > 0, index=data.index)
    sell = pd.Series(pacts < 0, index=data.index)
    # print(np.sum(buy-sell))
    OHLC = data[[('DealPrice', 'close'), ('DealPrice', 'close'), ('DealPrice', 'close'), ('DealPrice', 'close')]]
    OHLC.columns = ['O', 'H', 'L', 'C']
    # Основной график эквити
    bt = pb.Backtest(locals())
    return bt


load_model = True
model_path = 'model'
length = R.shape[0]
noise_level = 1e-5
price = R.values[:, 3]
n_noises = 1
n_runs = 1

test_acts = np.zeros((n_noises, n_runs, length - 1))
test_rews = np.zeros((n_noises, n_runs, length - 1))

for j in range(n_noises):
    np.random.seed(j)
    env = environment(R + (np.random.rand(R.shape[0], R.shape[1]) - 0.5) * (1e-5 * (0**j)), length)
    for i in range(n_runs):
        np.random.seed(i + 1377)
        test_acts[j, i, :], test_rews[j, i, :] = run_trades(
            max_episode_length, gamma, s_size, a_size, load_model, model_path, length, env)
        print(j, i)

# ПРИ НЕУВЕРЕННОСТИ НУЖНО ДЕЛАТЬ ПРЕДЫДУЩЕЕ ДЕЙСТВИЕ, А НЕ 0

test_acts_probs = np.zeros((n_noises, a_size, length - 1))
tresh = 0.0
acts = np.zeros((n_noises, length - 1))
test_acts = test_acts.astype(int)
for k in range(n_noises):
    for j in range(n_runs):
        for i in range(length - 1):
            a = test_acts[k, j, i]
            test_acts_probs[k, a + 1, i] += 1
    test_acts_probs[k, :, :] = test_acts_probs[k, :, :] / n_runs
    acts[k, :] = [np.argmax(x) - 1 if max(x) > tresh else 0 for x in test_acts_probs[k, :, :].T]
# print(test_acts_probs)
# plt.plot(acts.T)
# plt.show()


def write_report(r, filename):
    with open(filename, "a") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line, file=input_file)


folder = os.path.relpath(".", "..")
for i in range(1):
    lengths = [R.values.shape[0]]
    names = [' train on 3 months']
    bt = backtest(acts[-1, :lengths[i]], test_R[:lengths[i] + dep + 2])
    fig = plt.figure(figsize=(12, 10))
    plt.ylim(60000, 1400000)
    bt.plot_trades()
    plt.legend(['long enter', 'short enter', 'long exit', 'short exit', 'equity', 'price'], loc='best')
    plt.title(folder + names[i], fontname="Times New Roman")
    plt.ylabel('Rubles', fontname="Times New Roman")
    plt.xlabel('Time', fontname="Times New Roman")
    plt.yscale('log')

    plt.plot()
    # plt.show()
    plt.savefig('../new_plots/' + folder + '_' + names[i] + '.pdf', bbox_inches='tight', format='pdf')

    # Отчет по системе
    B = bt.report
    write_report(B, '../new_plots/' + folder + '_train.txt')
