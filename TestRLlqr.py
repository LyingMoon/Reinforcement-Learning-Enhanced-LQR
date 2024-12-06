import csv
import torch
from NNnetwork import PolicyNet  
import scipy.io
import numpy as np
from RLQubeModelVar import *
from QubeModel import *
from math import *
from matplotlib import pyplot as plt


n_states=5
n_hiddens=128
n_actions=1

lqr_action_bound=12.0
K = np.array([-10.0000000000000,47.3909009321752,-2.95417400084810,3.95689570879886])
rllqr_action_bound = 10.0
rl_action_bound = 2.0
Time=[]
frequency=100
state=np.array([[0],[0],[0],[0]])
rlstate = state
Qube = DiscreteTimeVarQubeModel(state)
RLQube = DiscreteTimeVarQubeModel(state)
#Qube = DiscreteTimeQubeModel(state)

rl = PolicyNet(n_states, n_hiddens, n_actions, rl_action_bound)
rl.load_state_dict(torch.load('case_1.pth'))
rl.eval()

x1=[]
x2=[]
x3=[]
x4=[]
rlx1=[]
rlx2=[]
rlx3=[]
rlx4=[]
refsignal = []
v=[]
rlv=[]
rewardlqr=0
rlrewardlqr = 0
weight = 1.2
feq=1
# The simulation loop
for i in range(0,701):
    Time.append(i/frequency)
    x1.append(state[0,0])
    x2.append(state[1,0])
    x3.append(state[2,0])
    x4.append(state[3,0])
    
    rlx1.append(rlstate[0,0])
    rlx2.append(rlstate[1,0])
    rlx3.append(rlstate[2,0])
    rlx4.append(rlstate[3,0])
    
    # Signal 1: Sine Wave
    ref1 = weight*sin(((i)/frequency)*feq)
    refsignal.append(ref1)
        
    ref = np.array([[ref1],[0],[0],[0]])
    modelstate = np.array([[state[0,0]],[state[1,0]],[state[2,0]],[state[3,0]]])
    rlmodelstate = np.array([[rlstate[0,0]],[rlstate[1,0]],[rlstate[2,0]],[rlstate[3,0]]])
    
    lqr_voltage = K@(ref - modelstate)
    lqr_voltage = lqr_voltage[0]
    
    rl_lqr_voltage = K@(ref - rlmodelstate)
    rl_lqr_voltage = rl_lqr_voltage[0]
    
    
    if abs(lqr_voltage) > abs(lqr_action_bound):
        lqr_voltage = lqr_action_bound
        
    if abs(rl_lqr_voltage) > abs(rllqr_action_bound):
        rl_lqr_voltage = rllqr_action_bound
    
    rllqrstate=np.array([rlstate[0,0],rlstate[1,0],rlstate[2,0],rlstate[3,0],ref1])
    input_tensor = torch.tensor(np.array(rllqrstate), dtype=torch.float32)
    with torch.no_grad(): 
        output = rl(input_tensor)
    rlInput=output.squeeze().item()

    if rlInput > rl_action_bound:
        rlInput = rl_action_bound
        
    rl_lqr_voltage = rl_lqr_voltage + rlInput
    
    Qube.updateState(lqr_voltage)
    RLQube.updateState(rl_lqr_voltage)
    
    rewardlqr = rewardlqr-10*(state[0,0]-ref1)**2- 10*(state[1,0])**2\
        -0.1*(lqr_voltage**2)

    rlrewardlqr = rlrewardlqr-10*(rlstate[0,0]-ref1)**2- 10*(rlstate[1,0])**2\
        -0.1*(rl_lqr_voltage**2)
    
    v.append(lqr_voltage)
    rlv.append(rl_lqr_voltage)
    
    state=Qube.getState()
    rlstate = RLQube.getState()

    
print(rewardlqr)
print(rlrewardlqr)

fig, axs= plt.subplots(5)
axs[0].plot(Time,x1,Time,rlx1,Time,refsignal)
axs[1].plot(Time,x2,Time,rlx2)
axs[2].plot(Time,x3,Time,rlx3)
axs[3].plot(Time,x4,Time,rlx4)
axs[4].plot(Time,v,Time,rlv)
axs[0].set(xlabel= 'Time(s)', ylabel='x1')
axs[1].set(xlabel= 'Time(s)', ylabel='x2')
axs[2].set(xlabel= 'Time(s)', ylabel='x3')
axs[3].set(xlabel= 'Time(s)', ylabel='x4')
axs[4].set(xlabel= 'Time(s)', ylabel='v')
plt.show()
