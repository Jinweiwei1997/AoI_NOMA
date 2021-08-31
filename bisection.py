import  numpy as np


def bisection(h,g,BEnergy,AoI,M, v1=1,v2=1):
    #AoISum The average sum of processes at base station
    V=1 #Lyapnov drift value
    flat=1 #define H or T
    AoI_k=[x for x in AoI] #k slot AoI
    Amax = 1000
    Bmax = 0.0003
    sigma = 3.162277660168375*10**(-8)
    S=1.5  #is S/W in this environment,W=10,S=15
    AverSumAoI = 0 #Sum of AoI at base station
    theta =[] #never used 权重
    eta = 0.5 #gain loss
    P = 5.012
    LyaAoI=0
    LyaAoI1=0
    EnergyHarvest = [0 for i in range(len(M))] #amount of energy harvest
    BEnergy_k = [x for x in BEnergy]
    LyaBEnergy = 0 #calculate Battery Energy changed
    LyaBEnergy1=0
    B_Lya2 = 0
    BSum=0
    B_change=0
    AoI_change=0
    Trans_nodes = []
    Trans_nodes_h=[]
    flatT=0
    for i in range(len(M)):
        if M[i]==1:
            flatT=1
            break
    if flatT==0: #If the action is harvesting
        for j in range(len(M)):
            # calculate the Energy Harvested
            EnergyHarvest[j] = eta * P * g[j]
            B_next = BEnergy_k[j] + EnergyHarvest[j]
            if B_next >= Bmax:
                BEnergy_k[j] = Bmax
            else:
                BEnergy_k[j] += EnergyHarvest[j]
            # calculate the Sum of AoI
        for j in range(len(M)):
            if (AoI[j] < Amax):
                AoI_k[j] = AoI[j] + 1
            else:
                AoI_k[j] = Amax
        for j in range(len(M)):
            AverSumAoI += AoI_k[j]
        AverSumAoI /= len(M)
    else:    #trans
        Energy_Trans=[]
        for i in range(len(M)): #nodes need to trans package
            if M[i]==1:
                AoI_k[i]=1
                Trans_nodes.append(i)
                Trans_nodes_h.append(h[i])
            else:
                AoI_k[i]+=1
        list_in = np.argsort(Trans_nodes_h) #对所有的要发的数据包的信道增益进行排序
        for i in range(len(list_in)):
            Energy_noise=0
            for k in range(len(Energy_Trans)):
                Energy_noise+=(Energy_Trans[k]*Trans_nodes_h[list_in[k]])
            Energy_Trans_Node=(sigma+Energy_noise)/Trans_nodes_h[list_in[i]]*(2**S-1)
            Energy_Trans.append(Energy_Trans_Node)
        for i in range(len(list_in)): #发送数据，对所有的暂时电池遍历BEnergy_k
            BEnergy_k[Trans_nodes[list_in[i]]]-=Energy_Trans[i]
        for i in range(len(M)):
            if BEnergy_k[i]<0:
                return -100000000,BEnergy
        for i in range(len(M)):
            AverSumAoI+=AoI_k[i]
        AverSumAoI/=(len(M))
    for i in range(len(M)):
        LyaBEnergy += (BEnergy_k[i]-BEnergy[i])*(BEnergy_k[i]-BEnergy[i])
        LyaBEnergy1 += BEnergy_k[i]*BEnergy_k[i]-BEnergy[i]*BEnergy[i]
    for i in range(len(M)):
        LyaAoI +=(AoI_k[i]-AoI[i])*(AoI_k[i]-AoI[i])
        B_Lya2 += BEnergy_k[i]*BEnergy_k[i]
        BSum +=BEnergy_k[i]
        B_change+=(BEnergy_k[i]-BEnergy[i])
        AoI_change+=AoI[i]
        LyaAoI1 += AoI_k[i]*AoI_k[i]-AoI[i]*AoI[i]
    # if LyaBEnergy1<0:
    #     LyaBEnergy1=-LyaBEnergy1
    if LyaAoI1<0:
        LyaAoI1=-LyaAoI1
    LyaAoI2=AoI_change-AverSumAoI
    LyapnovDrift =-AverSumAoI +100*LyaBEnergy1
    #LyapnovDrift = -v1 * AverSumAoI  + v2 * B_change
    return LyapnovDrift,AverSumAoI,BEnergy_k,AoI_k



if __name__ == "__main__":
    h =([6.06020304235508 * 10 ** -5, 1.10331933767028 * 10 ** -5, 1.00213540309998 * 10 ** -4,1.21610610942759 * 10 ** -4])
    g =([6.06020304235508 * 10 ** -6, 1.10331933767028 * 10 ** -5, 1.00213540309998 * 10 ** -7,1.21610610942759 * 10 ** -6])
    M = ([0, 0, 1, 1])
    BEnergy =[4*10**-4,4*10**-4,4*10**-4,4*10**-4]
    AoI=([1,1,1,1])
    a = bisection(h,g,BEnergy,AoI,M)
    print(a)
'''
if __name__ == "__main__":
    N=10;Memory = 1024
    mem = MemoryDNN(net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )
    mlist=mem.decode(mem,5)
    print(mlist)
'''