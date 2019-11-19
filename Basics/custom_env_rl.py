import numpy as np
from PIL import Image
import cv2
import pickle
import time
SIZE = 10
episodes = 10000
move_penalty = -1
police_penalty = -100
gold_penalty = 50
show_every = 1000
learning_rate = 0.2
gamma = 0.9

# for coloring 
thief_key = 1
police_key = 2
gold_key = 3

# RGB color coding
d = {1:(255, 0, 0), 2:(0,255,0), 3:(0,0,255)}


class Grid:
  def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

  def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

  def action(self, choice):
    '''
    Gives us 8 total movement options. (0,1,2,3,4,5,6,7)
    '''
    if choice == 0:
      self.move(x=1, y=1)
    elif choice == 1:
      self.move(x=-1, y=-1)
    elif choice == 2:
      self.move(x=-1, y=1)
    elif choice == 3:
      self.move(x=1, y=-1)
    elif choice == 4:
      self.move(x=1,y=0)
    elif choice == 5:
      self.move(x=0, y=1)
    elif choice == 6:
      self.move(x=-1, y=0)
    elif choice == 7:
      self.move(x=0, y=-1)
  
  def move(self, x=False, y=False):
    if not x:
      self.x += np.random.randint(-1, 2)
    else:
      self.x += x
    if not y:
      self.y += np.random.randint(-1,2)
    else:
      self.y += y
      
    if self.x<0:
      self.x=0
    if self.x>=SIZE:
      self.x = SIZE-1
    if self.y<0:
      self.y=0
    if self.y>=SIZE:
      self.y = SIZE-1


q_table = {}
for a in range(-SIZE+1, SIZE):
  for b in range(-SIZE+1, SIZE):
    for c in range(-SIZE+1, SIZE):
      for d in range(-SIZE+1, SIZE):
        for e in range(-SIZE+1, SIZE):
          for f in range(-SIZE+1, SIZE):
            q_table[((a,b),(c,d),(e,f))]= [np.random.uniform(-8, 0) for i in range(8)]


for eps in range(episodes):
  print(eps)
  police1 = Grid()
  police2 = Grid()
  gold = Grid()
  thief = Grid()
  show = False
  if(eps%show_every==0):
    show = True
  
  
  for i in range(200):
    dstate = (police1-thief, police2-thief, gold-thief)
    action = np.random.randint(0,8)
    thief.action(action)
    if(thief.x==police1.x and thief.y==police1.y):
      reward = police_penalty
    elif(thief.x==police2.x and thief.y==police2.y):
      reward = police_penalty
    elif(thief.x==gold.x and thief.y==gold.y):
      reward = gold_penalty
    else:
      reward = move_penalty
    
    new_dstate = (police1.substract(thief), police2.subtract(thief), gold.subtract(thief))
    max_future_qval = np.max(q_table[new_dstate])
    current_qval = q_table[dstate][action]
    if reward == gold_penalty:
      new_qval = gold_penalty
    else:
      new_qval = (1 - learning_rate) * current_qval + learning_rate * (reward + gamma * max_future_qval)
    
    q_table[dstate][action] = new_qval
    
    print("reward=", reward)
    if(show):
      env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) # 3 is the number of channels for RGB image
      env[gold.x][gold.y] = d[gold_key]
      env[thief.x][thief.y] = d[thief_key]
      env[police1.x][police1.y] = d[police_key]
      env[police2.x][police2.y] = d[police_key]
      image = Image.fromarray(env, 'RGB')
      image = image.resize((300, 300))
      cv2.imshow("ENV", np.array(image))
      if reward == gold_penalty or reward == police_penalty:
        if cv2.waitKey(500) and 0xFF == ord('q'):
          break
      else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
          
          
    if reward == gold_penalty or reward == police_penalty:
      break



