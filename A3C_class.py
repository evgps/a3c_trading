# -*- coding: utf-8 -*-
from configs import EXTRA_DENSE, N_HIDDEN, DROPOUT, COOL_V, COOL_A, dep, gamma, training

# магическая константа 37 - число фич
s_size = 38 * dep
a_size = 3
model_path = 'model'
import os
import pandas as pd
import numpy as np
import scipy

import tensorflow as tf
import tensorflow.contrib.slim as slim
# %matplotlib inline
from random import choice
from trader_gym import environment
# %load_ext ipycache


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, s_size])
            if(EXTRA_DENSE):
                hidden = slim.fully_connected(slim.flatten(self.imageIn), N_HIDDEN, activation_fn=tf.nn.tanh)
            else:
                hidden = slim.flatten(self.imageIn)

            if(DROPOUT):
                rnn_in = tf.layers.dropout(
                    hidden,
                    rate=0.5,
                    noise_shape=None,
                    seed=None,
                    training=training,
                    name='drop1')
            else:
                rnn_in = hidden

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(rnn_in, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, N_HIDDEN])

            if(COOL_A):
                a_in = slim.fully_connected(slim.flatten(rnn_out), 32, activation_fn=tf.nn.tanh)
            else:
                a_in = rnn_out

            self.policy = slim.fully_connected(a_in, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            if(COOL_V):
                v_in = slim.fully_connected(slim.flatten(rnn_out), 32, activation_fn=tf.nn.tanh)
            else:
                v_in = rnn_out

            self.value = slim.fully_connected(v_in, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(0.01),
                                              biases_initializer=None)

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, env, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter('tb/train_' + str(self.number))

        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        self.actions = [-1, 0, 1]

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)
        # лол, это равно discounted_rewaeds - sefl.value_plus

        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver, dep):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                action_buffer = [0] * dep
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                s = np.concatenate((s, action_buffer))
                episode_frames.append(s)
                rnn_state = self.local_AC.state_init
                summary = tf.Summary()

                while d == False:
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                    feed_dict={self.local_AC.inputs: [s],
                                                               self.local_AC.state_in[0]: rnn_state[0],
                                                               self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    s1, r, d, _ = self.env.step(self.actions[a])
                    # Добавим новое действие в конец буфера
                    action_buffer[:] = np.concatenate((action_buffer[1:], [self.actions[a]]))
                    s1 = np.concatenate((s1, action_buffer))
                    if not d:
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
#                     Save history of boss actions

                    if len(episode_buffer) == 200 and d != True and episode_step_count != max_episode_length - 1:
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                if episode_count % 50 == 0 and self.name == 'worker_0':
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")

                mean_reward = np.mean(self.episode_rewards[-5:])
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_value = np.mean(self.episode_mean_values[-5:])

                summary.value.add(tag='env/shares', simple_value=float(self.env.n_shares))
                summary.value.add(tag='Perf/Act', simple_value=float(a))
                summary.value.add(tag='Perf/Episode_reward', simple_value=float(episode_reward))
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                print(episode_count, episode_reward)


class Test_Worker():
    def __init__(self, env, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.acts = []
        self.rewards = []
        self.summary_writer = tf.summary.FileWriter('tb/train_' + str(self.number))
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = env
        self.actions = [-1, 0, 1]
        self.prev_act = 1

    def work(self, max_episode_length, gamma, sess, coord, saver, dep, tresh):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            sess.run(self.update_local_ops)
            d = False
            episode_reward = 0
            s0 = self.env.reset()
            action_buffer = [0] * dep
            s = np.concatenate((s0, action_buffer))
            rnn_state = self.local_AC.state_init
            summary = tf.Summary()

            while d == False:
                a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                feed_dict={self.local_AC.inputs: [s],
                                                           self.local_AC.state_in[0]: rnn_state[0],
                                                           self.local_AC.state_in[1]: rnn_state[1]})
                # print("A",a_dist[0], a_dist)
                p = np.amax(a_dist[0])
                a = np.argmax(a_dist[0])
                # print('argmax',p,'max',a)
                if(p < tresh):
                    a = self.prev_act
                self.prev_act = a
                # print(a)
                # a = np.random.choice(a_dist[0],p=a_dist[0])
                # print("RC",a)
                # a = np.argmax(a_dist == a)
                # print("ARGMAX",a)
                s1, r, d, _ = self.env.step(self.actions[a])
                action_buffer[:] = np.concatenate((action_buffer[1:], [self.actions[a]]))
                s1 = np.concatenate((s1, action_buffer))
                if d:
                    s1 = s
                total_steps += 1
                episode_reward += r
                s = s1
                self.acts.append(self.actions[a])
                self.rewards.append(r)

            self.episode_reward = np.cumsum(self.rewards)
            # for i in range(len(self.actions)):
            #     summary.value.add(tag='test/a', simple_value=float(self.actions[i]))
            #     summary.value.add(tag='test/r', simple_value=float(self.rewards[i]))
            #     summary.value.add(tag='test/equity', simple_value=float(self.episode_reward[i]))
            #     self.summary_writer.add_summary(summary, i)
            #     self.summary_writer.flush()
            if self.name == 'worker_0':
                sess.run(self.increment)
            episode_count += 1
            print(total_steps, episode_reward)
