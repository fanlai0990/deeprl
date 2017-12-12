#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random as rd
from ddpg import DDPG
from ddpg_net import *
from numpy import *

MAX_STEP = 5
EPISODES = 5000

MAX_FLOWS = 9              # the number of visible flows
NUM_NODES = 5
MAX_PATHS = 3
GML = 'swan.txt'
DATA_TRACE = 'flows_swan.txt'

S_DIM, A_DIM = (3 * MAX_FLOWS + (NUM_NODES ** 2)), MAX_PATHS        # state and action dimension
n_states = S_DIM
n_actions = A_DIM
action_low_bound = 0.
action_high_bound = 1.

agent = DDPG(n_states, n_actions, action_low_bound, action_high_bound)

def run():
    plt.ion()
    
    avg_ep_r_hist = []
    total_fcts = []
    avg_ep_r = -0.
    total_rs = {}

    with open('normal.txt') as fin:
        lines = fin.readlines()
        for i, item in enumerate(lines):
            item = item.strip().split()
            total_rs[i] = [float(item[0]), float(item[1])]

    for episode in range(EPISODES):
        ep_step = 0
        env = NetEnv(GML, DATA_TRACE, MAX_FLOWS, MAX_PATHS)

        s = env.get_obser()
        buffer_s = []
        buffer_a = []
        buffer_s_ = []

        total_r = 0
        len_ = 0

        while True:
            a = agent.choose_action(s)
            s_, r, done = env.step(a)

            buffer_s.append(s)
            buffer_s_.append(s_)
            buffer_a.append(a)

            #agent.store_memory(s,a,r, s_)
            
            if r != 0:
                r = (r+7.)/7.
                for i in xrange(len(buffer_a)):
                    #r *= 0.8
                    agent.store_memory(buffer_s[i], buffer_a[i], r, buffer_s_[i])
                buffer_s_, buffer_a, buffer_s = [], [], []
            
            ep_step += 1

            if r > -100:
                total_r += r

            if agent.memory_counter >= agent.batch_size:
                agent.learn()

            s = s_
            if done == True:
                total_fcts.append(env.get_total_fct())
                print 'total fct: ....', total_fcts[-1]
                break
 
        if episode >= 10:

            if avg_ep_r == 0:
                avg_ep_r = total_r
            else:
                avg_ep_r = avg_ep_r * 0.9 + 0.1 * total_r

            avg_ep_r_hist.append(avg_ep_r)

            if episode % 20 == 0:
                print('Episode %d Avg Reward/Ep %s' % (episode, avg_ep_r))

        plt.cla()
        plt.plot(avg_ep_r_hist)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.pause(0.0001)

    plt.ioff()

    '''ff = open('normal.txt','a')
    ff.close()
    with open('normal.txt','w') as fout:
        for key in total_rs:
            arr = array(total_rs[key])
            fout.writelines(str(arr.mean())  + '\t' + str(arr.std()) +'\n')
    '''
    ff = open('fcts.txt','a')
    ff.close()

    with open('fcts.txt','w') as fout:
        for item in total_fcts:
            fout.writelines(str(item)  + '\n')

    ff = open('rewards.txt','a')
    ff.close()

    with open('rewards.txt','w') as fout:
        for item in avg_ep_r_hist:
            fout.writelines(str(item)  + '\n')

    plt.figure(2)
    plt.plot(total_fcts)
    plt.xlabel('Iteration')
    plt.ylabel('Total Flow Completion Time')
    plt.show()

if __name__ == '__main__':
    run()
