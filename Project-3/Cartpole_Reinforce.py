import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v0')
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size=env.observation_space.shape[0], h_size=16, action_size=env.action_space.n):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, h_size)
        self.fc2 = nn.Linear(h_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def perform_one_episode(st):
    saved_probs=[]
    rewards=[]
    state=st
    while True:
        action, log_probility = policy.act(state)
        saved_probs.append(log_probility)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    
    return saved_probs,rewards


def update_policy(rewards,saved_log_probs):
    gamma=1
    #discounts = [gamma**i for i in range(len(rewards)+1)]
    discounts=[]
    for i in range(len(rewards)+1):
        discounts.append(gamma*i)
        
    #R = sum([a*b for a,b in zip(discounts, rewards)])
    R=[]
    s=[]
    for a,b in zip(discounts,rewards):
        c=a*b
        s.append(c)
    R=sum(s)  
        
    policy_loss = []
    for log_prob in saved_log_probs:
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
        
    
    

def reinforce():
    scores_deque = deque(maxlen=100)
    scores = []
    for n_episode in range(1, 1001):
        savedLogProbs = []
        rewards = []
        state = env.reset()
        
        slp,r=perform_one_episode(state)
        savedLogProbs = savedLogProbs + slp
        rewards=rewards+r
             
        sumOfRewards = sum(rewards)
        scores_deque.append(sumOfRewards)
        scores.append(sumOfRewards)
        
        update_policy(rewards,savedLogProbs)
        
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(n_episode-100, np.mean(scores_deque)))
            break
        
    return scores
    
scores = reinforce()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()







