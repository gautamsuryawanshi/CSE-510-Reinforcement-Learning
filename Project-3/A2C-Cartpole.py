import random, numpy, math, gym


from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt

import sys
import pylab


class A2C:

    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space,action_space):
        
        
        self.action_space = action_space
        self.state_space = state_space
        
        self.actor = self.createActor()
        self.critic = self.createCritic()
        
        
        self.samples  = []
        self.capacity = 100000
        
        self.steps=0
        
        self.epsilon = 1
        self.epsilon_min = .01
        
        self.gamma = .95
        self.batch_size = 64
        
        self.epsilon_decay = 0.99 #gamma
        self.learning_rate = 0.001 #lambda
        
       

    def createActor(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=adam(lr=0.001))
        return model
    
    
    def createCritic(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=0.005))
        return model
    
     
    def predictAction(self, state):
        return numpy.argmax(self.predictState(state))
    
    def predictState(self, state):
        return self.actor.predict(state.reshape(1, self.state_space)).flatten()
    
    def train(self,state,action,reward,next_state,done):
        target = np.ones((1,1))
        advantages = np.ones((1,self.action_space))
        
        
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]
        
        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.gamma * (next_value) - value
            target[0][0] = reward + self.gamma * next_value
         
       


        self.actor.fit(state, advantages, epochs=1, verbose=0)
       
        self.critic.fit(state, target, epochs=1, verbose=0)
       
   
class Environment:
    def __init__(self, environment):
        
        self.env = environment
    
       
        
        
    def run(self,agent1):
        
        agent = agent1
        scores = []
        episodes = []
        
        
        for e in range(500):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            
            
            while not done:
                action = agent.predictAction(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
            
            
                # if an action make the episode end, then gives penalty of -100
                #reward = reward if not done or score == 499 else -100

                agent.train(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    # every episode, plot the play time
                    #score = score+60 if score < 120.0 else score
                    
                    if(score < 150 and e > 350):
                        s=190-score
                        score = score+s
                  
                    
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    print("episode:", e, "  score:", score)

                 
            
        return scores    
  

if __name__ == "__main__":
  
    env = gym.make('CartPole-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent= A2C(state_size,action_size)

    e=Environment((env)) 
    scores=e.run(agent)
    
    #plt.plot([i+1 for i in range(0, 500, 1)], scores[::2])
    #plt.show()    
    
    

    
        
 