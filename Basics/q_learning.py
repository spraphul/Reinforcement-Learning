import numpy as np
import gym

env = gym.make("MountainCar-v0")

num_actions = env.action_space.n
dis_obs_space_size = [30]*len(env.observation_space.high) 
# you can tweak the dis_obs_space as  rectangular patch also, here its [30,30]
dis_obs_space_winsize = (env.observation_space.high-env.observation_space.low)/dis_obs_space_size

learning_rate = 0.2
episodes = 4000
gamma = 0.9

def cont2Discrete(state): 
   dstate = (state-env.observation_space.low)/dis_obs_space_winsize
   return tuple(dstate.astype(np.int))

q_table = np.random.uniform(low=-2, high=0, size=(dis_obs_space_size + [num_actions])) # a 3-D array 

show_every = 1000

for eps in range(episodes):
    print(eps)
    show = False
    done = False
    dstate = cont2Discrete(env.reset())
    if(eps%show_every==0):
        show = True
    
    
    
    
    while not done:
        action = np.argmax(q_table[dstate])
        new_state, reward, done, _ = env.step(action)
        new_dstate = cont2Discrete(new_state)
        
        if not done:
          current_qval = q_table[dstate + (action,)]
          max_future_qval = np.max(q_table[new_dstate])
          
          new_qval = (1-learning_rate)*current_qval + learning_rate*(reward+ gamma*max_future_qval)
          
          q_table[dstate+(action,)] = new_qval
        
        elif new_state[0]>=env.goal_position:
          q_table[dstate+(action,)]=0 # 0 is the reward
          
          
        dstate = new_dstate
        if(show):
           env.render()
    

env.close()
