# Human-level control through deep reinforcement learning [https://www.nature.com/articles/nature14236]
# Deep Reinforcement Learning with Double Q-learning [https://arxiv.org/abs/1509.06461]

import os
import random
import shutil
import argparse
import numpy as np
import tensorflow as tf
from env import make_env
from tensorflow.contrib import layers

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Breakout', help='Name of the game')
parser.add_argument('--seed', type=int, default=123, help='Which seed to use')
args = parser.parse_args()

#Network dependant arguments
conv1_filter_size = 8
conv1_layers = 18
conv1_stride_size = 4
conv2_filter_size = 4
conv2_layers = 32
conv2_stride_size = 2
fc1_layers = 256
fc2_layers = 4


#YOUR CODE HERE
#You must define these constants and set their value
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000000
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
DISCOUNT_FACTOR = 0.99
learning_rate = 0.0001
# learning_rate_minimum = 0.00025
# learning_rate_decay = 0.96
# learning_rate_decay_step = 50000


INITIAL_EXPLORATION = 1.0
UPDATE_FREQUENCY = 4
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
REPLAY_START_SIZE = 50000
ERROR_CLIP = 1
SAVE_FREQ = 50000
# Command-line Arguments
ENV_ID = args.env
SEED = args.seed
# Optimizer Parameters


BASE_DIR = os.path.join('./games', ENV_ID)
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
EPOCHS_DIR = os.path.join(BASE_DIR, 'epochs')
MONITOR_DIR = os.path.join(BASE_DIR, 'monitor')

shutil.rmtree(LOGS_DIR, ignore_errors=True)
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(EPOCHS_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)

env = make_env(ENV_ID)
#IF YOU WANT TO SEE YOUR AGENT ONLINE YOU NEED TO CALL env.render() AT EACH FRAME IN AGENT MAIN LOOP BUT IT WILL DROP THE PERFORMANCE
env.unwrapped.seed(SEED)
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Network:
    def __init__(self, scope, num_actions):
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.uint8, [None, 84, 84, 4])
            self.inorm = tf.cast(self.input, tf.float32) / 255.0
            self.actions = tf.placeholder(tf.int32, [None])
            self.target_q = tf.placeholder(tf.float32, [None])

            #YOUR CODE HERE
            #You must create the networks to output the q value the normalized input is in self.inorm
            #you must have hese variables:
            
            self.conv1_filter = tf.Variable(tf.random_normal([conv1_filter_size, conv1_filter_size, 4, conv1_layers], name = "conv1_filter"))
            self.bias_conv1 = tf.Variable(tf.fill([conv1_layers], 0.01, name = "bias_conv1"))
            self.conv1 = tf.nn.conv2d(self.inorm, 
                                self.conv1_filter,
                                strides = [1, conv1_stride_size, conv1_stride_size, 1], 
                                 padding="SAME", 
                                 name = "conv1") + self.bias_conv1
            self.conv1_relu = tf.nn.relu(self.conv1, name = "conv1_relu")
            
            self.conv2_filter = tf.Variable(tf.random_normal([conv2_filter_size, conv2_filter_size, conv1_layers, conv2_layers], name = "conv2_filter"))
            self.bias_conv2 = tf.Variable(tf.fill([conv2_layers], 0.01, name = "bias_conv2"))
            self.conv2 = tf.nn.conv2d(self.conv1_relu, 
                                self.conv2_filter,
                                strides = [1, conv2_stride_size, conv2_stride_size, 1],
                                padding="SAME",
                                name = "conv2") + self.bias_conv2
            self.conv2_relu = tf.nn.relu(self.conv2, "conv2_relu")
            self.flatten = tf.contrib.layers.flatten(self.conv2_relu)
            
            self.fc1 = tf.layers.dense(self.flatten, fc1_layers, activation = tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, fc2_layers)
            
            
            self.output_q = self.fc2
            self.action_q = tf.argmax(self.output_q, axis = 1)
            self.actions_onehot = tf.one_hot(self.actions, fc2_layers, on_value=1.0, off_value=0.0)

            self.max_q = tf.reduce_max(self.output_q, axis=1)
            self.q = tf.reduce_sum(self.output_q * self.actions_onehot, axis=1)

            #YOUR CODE HERE
            #YOU MUST DEFINE YOUR LOSS AND Optimizer
            #YOU MUST CREATE A self.update VARIBLE AND MINIMIZE YOUR LOSS WITH IT FOR EXAMPLE
            self.delta = self.q - self.target_q
            self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                        0.5 * tf.square(self.delta),
                                        tf.abs(self.delta) - 0.5, name='clipped_error')
            self.loss = tf.reduce_mean(self.clipped_error)
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("output_q", tf.reduce_sum(self.output_q, axis = 0))
            self.merged = tf.summary.merge_all()
  
            self.update = tf.train.RMSPropOptimizer(learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)


class MemoryBuffer:
    def __init__(self, buffer_size):
        self.n = buffer_size
        self.buffer = []
        self.p = 0

    def add(self, xp):
        if len(self.buffer) != self.n:
            self.buffer.append(None)
        self.buffer[self.p] = xp
        self.p = (self.p + 1) % self.n

    def sample(self, size):
        xp_list = random.sample(self.buffer, size)
        for i in range(size):
            xp = list(xp_list[i])  # shallow copy
            xp[0] = np.array(xp[0], copy=False)
            xp[3] = np.array(xp[3], copy=False)
            xp_list[i] = xp
        return np.array(xp_list, copy=False)


def copy_operation(src_scope, dst_scope):
    #YOUR CODE HERE
    #YOU MUST COPY VARIABLES IN SRC_SCOPE TO VARIABLES IN DST_SCOPE AND RETURN AN OPERATION ARRAY WHICH INCLUDE ASSIGNMENT TENSOR
    main_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=src_scope) 
    target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst_scope)

    assign_ops = []
    for main_var, target_var in zip(main_variables, target_variables):
        assign_ops.append(tf.assign(target_var, tf.identity(main_var)))
        
    return tf.group(*assign_ops)

tf.reset_default_graph()
replay_mem = MemoryBuffer(REPLAY_MEMORY_SIZE)
main_net = Network('main', env.action_space.n)
target_net = Network('target', env.action_space.n)
update_target = copy_operation('main', 'target')
saver = tf.train.Saver()
eps = INITIAL_EXPLORATION
delta_eps = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME
episode_t = 0
update_t = 0
epoch_t = 0
step_t = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(update_target)
    q_list = []
    r_sum = 0
    score = 0
    while True:
        episode_t += 1
        s = env.reset()
        while True:
            step_t += 1
            if random.random() < eps:
                a = random.randrange(env.action_space.n)
                [q] = sess.run(main_net.max_q, feed_dict={
                    main_net.input: [np.array(s, copy=False)]
                })
            else:
                [a], [q] = sess.run([main_net.action_q, main_net.max_q], feed_dict={
                    main_net.input: [np.array(s, copy=False)]
                })
            q_list.append(q)
            s_, r, d, i = env.step(a)
            r_sum += r
            score += i['real_reward']
            replay_mem.add([s, a, r, s_, d])
            if step_t > REPLAY_START_SIZE:
                if eps > FINAL_EXPLORATION:
                    eps -= delta_eps
                else:
                    eps = FINAL_EXPLORATION
                if step_t % UPDATE_FREQUENCY == 0:
                    update_t += 1
                    train_batch = replay_mem.sample(MINIBATCH_SIZE)
                    a = sess.run(main_net.action_q, feed_dict={
                        main_net.input: np.stack(train_batch[:, 3])
                    })
                    q = sess.run(target_net.q, feed_dict={
                        target_net.input: np.stack(train_batch[:, 3]),
                        target_net.actions: a
                    })
                    q *= 1.0 - train_batch[:, 4].astype(dtype=np.float32, copy=False)
                    target_q = train_batch[:, 2] + (DISCOUNT_FACTOR * q)
                    sess.run(main_net.update, feed_dict={
                        main_net.input: np.stack(train_batch[:, 0]),
                        main_net.actions: train_batch[:, 1],
                        main_net.target_q: target_q
                    })
                    if update_t % SAVE_FREQ == 0:
                        epoch_t += 1
                        epoch_str = str(epoch_t).zfill(3)
                        epoch_dir = os.path.join(EPOCHS_DIR, epoch_str)
                        os.makedirs(epoch_dir, exist_ok=True)
                        saver.save(sess, epoch_dir + '/model')
                if step_t % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    sess.run(update_target)
            s = s_
            if i['real_done']:
                #YOUR CODE HERE
                #YOU CAN ADD YOUR SUMMARIES HERE
                summary = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=score)])
                writer.add_summary(summary, episode_t)
                q_list = []
                r_sum = 0
                score = 0
            if d:
                break
