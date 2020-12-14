import random, numpy, math, gym


from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space,action_space):
        
        
        self.action_space = action_space
        self.state_space = state_space
        self.model = self.createModel()
         
        self.samples  = []
        self.capacity = 100000
        
        self.steps=0
        
        self.epsilon = 1
        self.epsilon_min = .01
        
        self.gamma = .95
        self.batch_size = 64
        
        self.epsilon_decay = .995  #gamma
        self.learning_rate = 0.001 #lambda
        
       

    def createModel(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=0.00025))
        return model
    
    def predictState(self, state):
        return self.model.predict(state.reshape(1, self.state_space)).flatten()
    
    def sample(self,n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
        

    def remember(self,sample):
        self.samples.append(sample)
        
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
        
        self.steps += 1
        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-self.learning_rate * self.steps)
        return self.epsilon
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space-1)
        else:
            return numpy.argmax(self.predictState(state))

    def replay(self):
        #batch = self.sample(self.batch_size)
        batch= random.sample(self.samples, min(self.batch_size, len(self.samples)))
        
        batchLen = len(batch)
        
        no_state = numpy.zeros(self.state_space)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        predictedStates = self.model.predict(states)
        predictedNextStates = self.model.predict(states_)

        x = numpy.zeros((batchLen, self.state_space))
        y = numpy.zeros((batchLen, self.action_space))
        
        for i in range(batchLen):
            o = batch[i]
            state = o[0]; 
            action = o[1]; 
            reward = o[2]; 
            nextState= o[3]
            
            target = predictedStates[i]
            if nextState is None:
                target[action] = reward
            else:
                target[action] = reward + self.epsilon_decay * numpy.amax(predictedNextStates[i])
                
      
            x[i] = state
            y[i] = target

        self.model.fit(x, y, batch_size=64, nb_epoch=1, verbose=0)
        
        
            

class Environment:
    def __init__(self, environment):
        self.environment = environment
        self.env = gym.make(environment)
        self.R = None
    def run(self, agent):
        state = self.env.reset()
        self.R=0

        while True:            
            self.env.render()

            action = agent.act(state)

            nextState, reward, done, info = self.env.step(action)

            if done: # terminal state
                nextState = None

            eDecay=agent.remember( (state, action, reward, nextState) )
            agent.replay()            

            state = nextState
            self.R = self.R + reward

            if done:
                break

        print("Total reward:", self.R)
        return self.R,eDecay





env = Environment('CartPole-v0')

states  = env.env.observation_space.shape[0]
actions = env.env.action_space.n

print(actions)
print(states)

agent = DQN(states, actions)

reward=[]
eDecay=[]

episodes=2000
for i in range(episodes):
    r,e=env.run(agent)
    reward.append(r)
    eDecay.append(e)
    

plt.plot([i+1 for i in range(0, episodes, 2)], reward[::2])
plt.show()    
plt.plot([i+1 for i in range(0, episodes, 2)], eDecay[::2])
plt.show()    