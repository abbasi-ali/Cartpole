# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:05:26 2021

@author: sociaNET_User
"""
import gym 
import numpy as np 
import torch 
from collections import deque
import random
import torch.optim as optim 
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import torch.nn as nn 
from tqdm import tqdm 
import cartpole_2


class CartNet(nn.Module):
    def __init__(self):
        super(CartNet, self).__init__()
        self.layer1 = nn.Linear(4, 164)
        # self.d1 = nn.Dropout(0.8)
        self.layer2 = nn.Linear(164, 2)
        # self.d2 = nn.Dropout(0.8)
        # self.layer3 = nn.Linear(256, 512)
        # self.d3 = nn.Dropout(0.8)
        # self.layer4 = nn.Linear(512, 256)
        # self.d4 = nn.Dropout(0.8)
        # self.layer5 = nn.Linear(256, 128)
        # self.d5 = nn.Dropout(0.8)
        # self.layer6 = nn.Linear(128, 2)
        
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        # x = self.d3(F.relu(self.layer3(x)))
        # x = self.d4(F.relu(self.layer4(x)))
        # x = self.d5(F.relu(self.layer5(x)))
        # x = self.layer6(x)
        
        return x

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

env = gym.make('CartPole-v0')

# high = torch.from_numpy(env.high).to(device)
# low = torch.from_numpy(env.low).to(device)


# chunk_p = 20 
# chunk_v = 20 

action_num = 2
episodes = 10


model = CartNet().to(device)
model.load_state_dict(torch.load('cart3.pt'))

for episode in tqdm(range(episodes)):
    done = False
    s = env.reset()
    
    s = torch.from_numpy(s).unsqueeze(0).float().to(device)
    
    step = 0    
    while done==False:


        
        sel_a = torch.argmax(model(s)).item()
            
        
        s, r, done, _ = env.step(sel_a)
        s = torch.from_numpy(s).unsqueeze(0).float().to(device)
        # s_prime = torch.from_numpy(s_prime).unsqueeze(0).float().to(device)
    
        env.render()
        
env.close()




