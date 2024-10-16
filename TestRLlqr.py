import csv
import torch
from NNnetwork import PolicyNet  
import scipy.io
import numpy as np
from QubeModel import *
from math import *
from matplotlib import pyplot as plt

n_states=5
n_hiddens=128
n_actions=1

lqr_action_bound=15.0

Time=[]
frequency=100
state=np.array([[0.5],[0],[0],[0]])

Qube = DiscreteTimeQubeModel(state)

x1=[]
x2=[]
x3=[]
x4=[]

v=[]
rewardlqr=0

weight = 1
feq=2
# The simulation loop
for i in range(0,701):
    Time.append(i/frequency)
    x1.append(state[0,0])
    x2.append(state[1,0])
    x3.append(state[2,0])
    x4.append(state[3,0])
    

    # Signal 1: Sine Wave
    ref1 = weight*sin(((i)/frequency)*feq)
   
    K = np.array([-10.0000000000000,47.3909009321752,-2.95417400084810,3.95689570879886])
        
    ref = np.array([[ref1],[0],[0],[0]])
    modelstate = np.array([[state[0,0]],[state[1,0]],[state[2,0]],[state[3,0]]])
    
    lqr_voltage = K@(ref - modelstate)
    lqr_voltage = lqr_voltage[0]
    
    if abs(lqr_voltage) > abs(lqr_action_bound):
        lqr_voltage = lqr_action_bound
        
    Qube.updateState(lqr_voltage)
    
    rewardlqr = rewardlqr-10*(state[0]-weight*sin(i/frequency*feq))**2-10*(state[1])**2\
        -0.1*(lqr_voltage**2)

    v.append(lqr_voltage)
    state=Qube.getState()

    
print(rewardlqr)

fig, axs= plt.subplots(5)
axs[0].plot(Time,x1)
axs[1].plot(Time,x2)
axs[2].plot(Time,x3)
axs[3].plot(Time,x4)
axs[4].plot(Time,v)
axs[0].set(xlabel= 'Time(s)', ylabel='x1')
axs[1].set(xlabel= 'Time(s)', ylabel='x2')
axs[2].set(xlabel= 'Time(s)', ylabel='x3')
axs[3].set(xlabel= 'Time(s)', ylabel='x4')
axs[4].set(xlabel= 'Time(s)', ylabel='v')
plt.show()
