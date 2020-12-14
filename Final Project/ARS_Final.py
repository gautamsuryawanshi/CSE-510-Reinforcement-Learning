
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
import matplotlib.pyplot as plt

#The HyperParameters Required

total_steps = 1000
episode_length = 1000
learning_rate = 0.02
total_directions = 16
good_directions = 16
noise = 0.03

env_name = 'HalfCheetahBulletEnv-v0'



class Normalizer():
    
    def __init__(self, input_space):
        self.n = np.ones(input_space)
        self.mean = np.ones(input_space)
        self.mean_diff = np.ones(input_space)
        self.var = np.ones(input_space)
    
    def observe(self, x):
        self.n = self.n + 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        res = (inputs - obs_mean) 
        return (res/ obs_std)

# Building the AI

class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.ones((output_size, input_size))
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        
        val=noise*delta
        if direction == "positive":
            return (self.theta + val).dot(input)
        else:
            return (self.theta - val).dot(input)
    
    def sample_deltas(self):
        
        sample=[]
        for i in range(total_directions):
            sample.append(np.random.randn(*self.theta.shape))
        
        return sample

            
    def update(self, r_pos, r_neg, d, sigma_r):
        step = np.ones(self.theta.shape)
   
        for i in range(total_directions):
            s = (r_pos[i] - r_neg[i])*d[i]
            step = step + s
        
        val = learning_rate / (good_directions * sigma_r) * step
        self.theta = self.theta + val


def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    numTrials = 0.
    total_rewards = 0
    while not done and numTrials < episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        total_rewards += reward
        numTrials += 1
    return total_rewards


    

def train(env, policy, normalizer):
    totalR=[]
    
    for step in range(total_steps):
        # initialize the random noise deltas and the positive/negative rewards
        
        deltas = policy.sample_deltas()
        positive_rewards = negative_rewards = [i for i in range(total_directions)]  
          
         # play an episode each with positive deltas and negative deltas, collect rewards
        for k in range(total_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
       
         # Compute the standard deviation of all rewards
        total_rewards = positive_rewards + negative_rewards
        all_rewards = np.array(total_rewards)
        sigma_r = np.std(all_rewards)
        
        
          # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
        scores ={}
        for s in range(total_directions):
            maxVal= max(positive_rewards[s],negative_rewards[s])
            scores.update({s:maxVal})
            
        
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:good_directions]
        
        pos_rew = []
        neg_rew = []
        delts = []
        
        for k in order:
            pos_rew.append(positive_rewards[k])
            neg_rew.append(negative_rewards[k])
            delts.append(deltas[k])
            
            
         # Update the policy    
        policy.update(pos_rew, neg_rew,delts,sigma_r)
        
        reward_evaluation = explore(env, normalizer, policy)
        totalR.append(reward_evaluation)
        
        print('Step:', step, 'Reward:', reward_evaluation)
        
    plotResults(totalR)
    



def saveVideo(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plotResults(totalR):
    fig = plt.figure()
    plt.plot(np.arange(1, len(totalR)+1), totalR)
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()


def runEnv(env):
    inputs_space = env.observation_space.shape[0]
    outputs_space = env.action_space.shape[0]  
    policy = Policy(inputs_space, outputs_space)
    normalizer = Normalizer(inputs_space)
    train(env, policy, normalizer)
    

env = gym.make(env_name)

#env = wrappers.Monitor(env, monitor_dir, force = True)
#work_dir = mkdir('exp', 'brs')
#monitor_dir = mkdir(work_dir, 'monitor')

runEnv(env)
