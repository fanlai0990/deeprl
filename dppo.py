"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import threading
import Queue as queue
import os, time
from ddpg_net import *
from numpy import *

EP_MAX = 10000
EP_LEN = 100
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
BETA = 0.5
ALPHA = 0.7
A_LR = 0.0002               # learning rate for actor
C_LR = 0.0005               # learning rate for critic
MIN_BATCH_SIZE = 32         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps, episodes
EPSILON = 0.2               # for clipping surrogate objective
F_DIM = 3                   # features of each flow: size, deadline, src-dest id
NORMALIZE_FACTOR = 10.0

MAX_FLOWS = 10              # the number of visible flows
NUM_NODES = 5
MAX_PATHS = 3
CONCAT_TOPO = 64
CONCAT_FLOWS = 128           # scale to F_DIM * MAX_FLOWS

S_DIM, A_DIM = (3 * MAX_FLOWS + (NUM_NODES ** 2)), MAX_PATHS        # state and action dimension

TOPO_DIM = NUM_NODES ** 2
ID_DIM = MAX_FLOWS
SIZE_DIM = MAX_FLOWS

GML = 'swan.txt'
DATA_TRACE = 'flows_swan.txt'
model_dir = "mnist"
model_name = "ckp_" + str(GAMMA)

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        index = NUM_NODES ** 2

        # treat the network topology as a black box. 
        # now = tf.layers.dense(tf.slice(self.tfs, [0, 0], [-1, 3]), 3, tf.nn.tanh)
        topologyl1 = tf.layers.dense(tf.slice(self.tfs, [0, 0], [-1, index]), CONCAT_TOPO, tf.nn.relu)
        #topologyl2 = tf.layers.dense(topologyl1, 32, tf.nn.relu)

        # construct the scheduling set
        flow_l1 = tf.layers.dense(tf.slice(self.tfs, [0, index], [-1, -1]), CONCAT_FLOWS, tf.nn.relu)     # id
        
        self.tfss = tf.concat([topologyl1, flow_l1], 1)
        l1 = tf.layers.dense(self.tfs, 128, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)  # pi = action distribution
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0) 
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5) # KL distance
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        '''
        try:
            self.saver.restore(self.sess, os.path.join(model_dir, model_name))
            print 'successfully load!'
        except Exception, e:
            self.sess.run(self.init)
            print e
        '''
        self.sess.run(self.init)

    def update(self):
        global GLOBAL_UPDATE_COUNTER

        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):

        with tf.variable_scope(name):

            l1 = tf.layers.dense(self.tfs, 128, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l1, A_DIM, tf.nn.sigmoid, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, 0., 1.)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save_model(self):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        self.saver.save(self.sess, os.path.join(model_dir, model_name))
        print "successfully restore!"

class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.ppo = GLOBAL_PPO
        self.DATA_MEAN = -17.45
        self.DATA_STD = 6

    def normalize(self, score):
        return (score - self.DATA_MEAN)/(self.DATA_STD + 1e-6)

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, GAMMA

        total_r = []
        global_r = 0.
        LOCAL_EP = 0

        while not COORD.should_stop():  # return true when thread stops
            env = NetEnv(GML, DATA_TRACE, MAX_FLOWS, MAX_PATHS)
            start_t = time.clock()
            buffer_s, buffer_a, buffer_r, buffer_s_,buffer_s_k = [], [], [], [], []
            s = env.get_obser()

            # several schedules
            while True:
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated

                a = self.ppo.choose_action(s)

                s_, r, done = env.step(a)

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_s_.append(s_)
                buffer_s_k.append(1)

                if r != 0:
                    r = (r+10)/10.
                    for i in xrange(len(buffer_s_k)):
                        buffer_r.append(r)                    # normalize reward, find to be useful

                    buffer_s_k = []

                s = s_
                global_r += r

                if r != 0:

                    GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                    discounted_r = []                           # compute discounted reward
                    len_ = len(buffer_s_) - 1

                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * self.ppo.get_v(buffer_s_[len_])
                        discounted_r.append(v_s_)
                        len_ -= 1

                    discounted_r.reverse()

                    #print buffer_s
                    #print buffer_r

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                if done == True:
                    LOCAL_EP += 1
                    break
        
            print 'global_ep: ', GLOBAL_EP, '\n'
            print 'simulation finished.....', env.get_total_fct(),'\n'
            #with open('result.txt','a') as out:
            #    out.writelines(str(self.wid) + ' finished with time: ' + str(env.get_total_fct()) + '\treward: ' + str(global_r) + '\n')
            
           # print 'simulation time cost....', time.clock() - start_t
            if GLOBAL_EP >= EP_MAX:         # stop training
                COORD.request_stop()
                break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: 
                GLOBAL_RUNNING_R.append(global_r)
            else: 
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+global_r*0.1)

            GLOBAL_EP += 1
            if GLOBAL_EP >= 10:
                print('{0:.1f}%'.format(GLOBAL_EP/double(EP_MAX)*100), '|W%i' % self.wid,  '|Ep_r: ' + str(global_r/double(LOCAL_EP)))

            #if GLOBAL_UPDATE_COUNTER % 100 == 0:
            #    self.ppo.save_model()

if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    #plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    #plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
            