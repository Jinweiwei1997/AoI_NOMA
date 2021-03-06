#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function

import operator
import cProfile
import time

from torch.autograd.grad_mode import F

from  bisection import bisection
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# from bisection import bisection

print(torch.__version__)


# DNN network for memory
class MemoryDNN():
    def __init__(
        self,
        net,
        learning_rate = 0.001,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):
        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1
        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.BatchNorm1d(self.net[3]),
                nn.Sigmoid(),
        )
    def _build_net_1(self):
        self.model = nn.Sequential(
                nn.Conv1d(1,16,3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16,16,3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16,1,1),
                nn.Linear(16, self.net[3]),
                 #nn.BatchNorm1d(self.net[3]),
                nn.Sigmoid()
             )
    #     self.context_net=nn.Sequential(
    #         nn.Conv1d(1, 16,3),
    #         nn.BatchNorm1d(16),
    #         nn.ReLU(),
    #         nn.Conv1d(16,16,3),
    #         nn.BatchNorm1d(16),
    #         nn.ReLU(),
    #         nn.Conv1d(16,1,1)
    #     )
    #
    #     self.fc=nn.Sequential(
    #          nn.Linear(16, self.net[3]),
    #          #nn.BatchNorm1d(self.net[3]),
    #          nn.Sigmoid()
    #      )
    #
    # def forward(self, x):
    #    x=self.context_net(x)
    #    x=x.view(x.size(0),-1)
    #    print(x.shape)
    #    out=self.fc(x)

       #return out



    def remember(self, h,g,BEnergy,AoI, m):


        # replace the old memory with new memory

        idx = self.memory_counter % self.memory_size
        #assert (m[np.argmax(m)] == 1)
        self.memory[idx, :] = np.hstack((h,g,BEnergy,AoI, m))
        self.memory_counter += 1


    def encode(self, h, g,BEnergy,AoI,m):
        # encoding the entry
        self.remember(h,g,BEnergy,AoI, m)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if  self.memory_counter % self.training_interval == 0 :
            self.learn()

    def learn(self):

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        # h_train=np.array(h_train)
        # h_train=h_train[np.newaxis,:]
        # h_train=torch.Tensor(h_train)
        # print('==============',h_train.shape)
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])
        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()
        self.cost = loss.item()
        #assert(self.cost > 0)
        self.cost_his.append(self.cost)
        '''
        if(self.memory_counter > 30000 and (self.memory_counter-30000)%128==0):
            train_acc = 0
            for x in range(128):
                flat = 0
                train_correct = 0

                if torch.Tensor.argmax((predict[x, :]))==np.argmax(m_train[x, :]):

                    train_correct = 1
                    train_acc += train_correct
            print("?????????",train_acc/128)
'''

    def decode(self, h, g, BEnergy, AoI,k=1, decoder_mode='OP'):
        # to have batch dimension when feed into Tensor
        #h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        temp = torch.Tensor([np.hstack((h, g, BEnergy, AoI))])
        #temp=temp[np.newaxis,:]
        # print('===', temp.shape)
        m_pred = self.model(temp)
        m_pred = m_pred.detach().numpy()

        if decoder_mode is 'OP':
            return self.knm(h, g, BEnergy, AoI,m_pred[0], k)+self.knm(h, g, BEnergy, AoI,m_pred[0]+np.random.normal(0,1,len(m_pred[0])), k)
        elif decoder_mode is 'KNN':
            return self.knn(h, g, BEnergy, AoI,m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")
    def opn(self, m, k= 1):
        return self.knm(m,k)+self.knm(m+np.random.normal(0,1,len(m)),k)
    def knm(self,h, g, BEnergy, AoI, m, k=1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ???rst binary of???oading decision with respect to equation (8)
        m_list.append(1 * (m > 0.5))
        if k > 1:
            # generate the remaining K-1 binary of???oading decisions with respect to equation (9)
            m_abs = abs(m)
            idx_list = np.argsort(m_abs)[:k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    # set a positive user to 0
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    # set a negtive user to 1
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self,h, g, BEnergy, AoI, m, k=1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[3]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


