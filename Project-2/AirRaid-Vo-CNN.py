import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import Conv2D


class DQN:

    def __init__(self, action_space):
        self.exploration_rate = 1

        
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model=self.createModel()
        
        self.gamma=0.95
        self.learningRate=0.001
        self.memorySize=100000
        self.batchSize=20
        self.explorationMax=1.0
        self.explorationMin=0.01
        self.explorationDecay=0.995
        
        
    def createModel(self):
        model1 = Sequential()
        model1.add(Conv2D(32,(3,3), activation="relu" ))
        model1.add(Flatten())
        model1.add(Dense(24, activation="relu"))
        model1.add(Dense(24, activation="relu"))
        model1.add(Dense(self.action_space, activation="linear"))
        model1.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model1
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        return np.argmax((self.model.predict(state))[0])
    
    

    def exp(self):
        self.exploration_rate *= self.explorationDecay
        self.exploration_rate = max(self.explorationMin, self.exploration_rate)
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
       
        
        for state, action, reward, state_next, terminal in batch:
            
            qUpdate = reward
            if not terminal:
                qUpdate = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            
            qValues = self.model.predict(state)
            
            qValues[0][action] = qUpdate
            
            self.model.fit(state, qValues, verbose=0)
        
        self.exp()
       


def cartpole(e,s,a):
    print("*********")
    env = e
    state_space = s
    action_space = a
    
    dqnObj = DQN(action_space)
    run = 0
    r=[]
    while run < 1000:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, state_space[0], state_space[1], state_space[2]])
        step = 0
        
        while True:
            step = step+1
            #env.render()
            action = dqnObj.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            r.append(reward)
            #print("Reward",reward)
            state_next = np.reshape(state_next, [1, state_space[0], state_space[1], state_space[2]])
            dqnObj.remember(state, action, reward, state_next, terminal)
            state = state_next
            
              
            if terminal:
                #print("*********")
                break
            dqnObj.replay()
    

if __name__ == "__main__":
    envName='AirRaid-v0'
    env = gym.make(envName)
    states= env.observation_space.shape
    actions= env.action_space.n
    cartpole(env,states,actions)