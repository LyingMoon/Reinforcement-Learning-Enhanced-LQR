import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from TrainedActorDDPG import ReplayBuffer, DDPG
from RLQubeModelVar import *
from NNnetwork import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# load the environment
# -------------------------------------- #

statemodel=np.array([[0],[0],[0],[0]])
env = DiscreteTimeVarQubeModel(statemodel)

n_states = 5  # number of states
n_actions = 1  # number of actions
n_hiddens = 128
action_bound = 2 # action bound
lqr_action_bound = 10
batch_size=64 # Train 64 samples for each time
sigma=0.2 # Gaussian Noise
tau=0.01
actor_lr=0.00005#5e-4
critic_lr=0.002#3e-3
gamma = 0.99
buffer_size=40000
min_size=10000
# -------------------------------------- #
# build the model
# -------------------------------------- #


# initialize the replay buffer
replay_buffer = ReplayBuffer(capacity=buffer_size)
# initialize the DDPG agent
agent = DDPG(n_states = n_states,  # Num of States
             n_hiddens = n_hiddens,  # hidden layers
             n_actions = n_actions,  # Num of Actions
             action_bound = action_bound,  # 
             sigma = sigma,  # Gaussian Noise
             actor_lr = actor_lr,  # actor learning rate
             critic_lr = critic_lr,  # critic learning rate
             tau = tau,  # soft update 
             gamma = gamma,  # gamma
             device = device
            )

# -------------------------------------- #
# training the model
# -------------------------------------- #

return_list = []  # rocord return for each 
mean_return_list = []  # average return
roll_list = []
pitch_list = []
yaw_list = []

for i in range(500):  # Iteration for n episode
    episode_return = 0  # culmulated for each episode chain
    weight=1#(np.random.rand()-0.5)*2
    state = env.reset(weight) # reset to original
    done = False  # the episode comes to an end
    time=0 # The Timer
    
    while not done and time<7:
        # what action it will take
        action = agent.take_action(state)
        input_tensor = torch.tensor(np.array(state), dtype=torch.float32)
        
        
        K = np.array([-10.0000000000000,47.3909009321752,-2.95417400084810,3.95689570879886])
        
        modelstate = np.array([[state[0,0]],[state[0,1]],[state[0,2]],[state[0,3]]])
        ref = np.array([[state[0,4]],[0],[0],[0]])
        
        lqr_voltage = K@(ref - modelstate)
        lqr_voltage = lqr_voltage[0]
        if abs(lqr_voltage) > abs(lqr_action_bound):
            lqr_voltage = lqr_action_bound
        
        
        actionAll = lqr_voltage + action
        # update the environment
        next_state, reward, done, _, _ = env.step(actionAll,time,weight) 
        # update the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        # update the state
        state = next_state
        # return
        episode_return = episode_return + reward
        # if the replay buffer reach its size for training, start training
        if replay_buffer.size() > min_size:
            # random sample batch_size
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            # create a dataset
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # train the model
            # Store the value
            agent.update(transition_dict)
        # update the time
        time=time+0.01
        
    # save the episode return for each 
    return_list.append(episode_return)
    mean_return_list.append(np.mean(return_list[-10:])) 
    
    # print the information for this episode
    print(f'iter:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}, weight: {weight}')

    if np.mean(return_list[-10:]) >= -25 and i>100:
        break
    
    
    
# Save the trained actor and critic
torch.save(agent.actor.state_dict(), 'TrainActor2.pth')  # Saves only the model parameters
torch.save(agent.target_actor.state_dict(), 'TargetActor2.pth') 
torch.save(agent.critic.state_dict(), 'TrainCritic2.pth') 
torch.save(agent.target_critic.state_dict(), 'TargetCritic2.pth') 


# -------------------------------------- #
# Plot
# -------------------------------------- #

x_range = list(range(len(return_list)))

plt.subplot(121)
plt.plot(x_range, return_list)  # return for every episode
plt.xlabel('episode')
plt.ylabel('return')
plt.subplot(122)
plt.plot(x_range, mean_return_list)  # average return
plt.xlabel('episode')
plt.ylabel('mean_return')
plt.show()

