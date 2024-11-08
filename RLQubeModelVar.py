import numpy as np
from math import *

class DiscreteTimeVarQubeModel:
    def __init__(self, initialState):
        
        self.state = np.array(initialState, dtype=float)
        # Define the A matrix
        self.A = np.array([[1,0.00728868025815964,0.00952581068107915,-7.14503247012065e-05],
                        [0,1.01285496974895,-0.000466765281370621,0.00987382929717613],
                        [0,1.42909299875816,0.906876461303208,-0.0115326590222530],
                        [0,2.53878104999700,-0.0915187622510064,0.979418895580092]])
        # Define the B matrix
        self.B = np.array([ [0.00365698703537935],
                            [0.00359973224707412],
                            [0.718176390977831],
                            [0.705800223015473]])

    def updateState(self, controlInput):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, np.array(controlInput, dtype=float))

    def getState(self):
        return self.state
    
    def step(self,controlInput,time,weight):
        Input=np.array(controlInput, dtype=float)
        Input=Input.T
        self.state = np.dot(self.A, self.state) + np.dot(self.B, np.array(Input, dtype=float))
        modelstate=self.state.T

        ref1 = np.array([weight*sin(time+0.01)]).reshape(1, -1)
        
        #Define the reward function here
        reward = -10 * ((self.state[0, 0] - ref1[0, 0])**2) \
                 -10 * ((self.state[1, 0])**2)\
                 -0.1 * ((Input[0, 0])**2)
        RLstate= np.concatenate((modelstate, ref1), axis=1)
        isDone=False
        if abs(self.state[0,0])>1.6 or abs(self.state[1,0])>0.4:
            isDone=True
            reward=reward-1000
        return RLstate,reward,isDone,False,{}
    
    def reset(self,weight):
        self.state = np.array([[0],[0],[0],[0]])
        modelstate=self.state.T
        ref1 = np.array([weight*sin(0)]).reshape(1, -1)
        RLstate= np.concatenate((modelstate, ref1), axis=1)
        return RLstate
    