import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
from bisection import  bisection
# Implementated based on the PyTorch
import torch

from memoryPyTorch import MemoryDNN


import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def plot_AoI(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    mpl.style.use('seaborn')
    plt.plot(np.arange(len(rate_array)) + 1, rate_his)
    plt.ylabel('Averagy Sum AoI')
    plt.xlabel('Time Frames')
    plt.show()
def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''
    N = 3                       # number of users
    n = 12000                    # number of time frames
    K = N+1                        # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 3200                   # Update interval for adaptive K
    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    data = sio.loadmat('./data/samechannel.mat')
    channel_h = data['input_h']
    channel_g = sio.loadmat('./data/samechannel.mat')['input_g']
    #start the defination
    AoI_t = [1 for x in range(N)]
    BEnergy_t = [0.0002 for x in range(N)]
    pl_AoI = []
    Amax = 100000
    Bmax = 0.003
    sigma = 3.162277660168375 * 10 ** (-9)
    S = 1.5
    theta = []  # never used 权重
    eta = 0.5  # gain loss
    P = 5.012
    EnerH=[0 for x in range(N)]
    EnerT=[0 for x in range(N)]
    AoI_F=[]
    flat = 1  #harvst or trans
    AverSumAoI=1
    AoIN=[]
    flat1 = 0  # have no energy
    k=0

    for i in range(3000):
        SumAoI=0
        flat1 = 0
        if flat==1:
            for j in range(N):
                EnerH[j]=eta*P*channel_g[i][j]
            for j in range(N):
                BEnergy_t[j]+=EnerH[j]
                if BEnergy_t[j]>Bmax:
                    BEnergy_t[j]=Bmax
            for j in range(N):
                AoI_t[j] += 1
                SumAoI += AoI_t[j]
            SumAoI /= N
            AverSumAoI = (AverSumAoI*i+SumAoI) /(i+1)
            flat = 0
            flat1=0
        else:
            for j in range(N):
                AoI_t[j] += 1
            for j in range(N):
                EnerT[j]=sigma/channel_h[i][j]*(2**S -1)
            if BEnergy_t[k] > EnerT[k]:
                BEnergy_t[k] -= EnerT[k]
                AoI_t[k] = 1
                flat1 = 1
            for j in range(N):
                SumAoI += AoI_t[j]
            SumAoI /= N
            AverSumAoI = (AverSumAoI * i + SumAoI) / (i + 1)
            if k==N-1 and flat1==0:
                flat=1
            k = ((k+1)%N)
        AoIN.append(AverSumAoI)
        AoI_F.append([x for x in AoI_t])
    print(AverSumAoI)
    save_to_txt(AoIN,"RRAOI")
    save_to_txt(AoI_F, "RRAOI1")
    plot_AoI(AoIN)
