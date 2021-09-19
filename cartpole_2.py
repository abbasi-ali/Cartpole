# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:45:23 2021

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
    
    
if __name__ == "__main__"   :
        
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
    episodes = 10000
    
    lr = 1
    gamma = 0.75
    eps = 0.9
    
    min_eps = 0.001
    start_decay = 1
    end_decay = 400
    decay_amt = (eps - min_eps) / (end_decay - start_decay)
    
    
    render_every = 10 
    copy_every = 5
    plot_every = 5
    
    mini_batch_size = 64 
    mem_size = 10_000
    
    epochs = 1
    
    model = CartNet().to(device)
    target = CartNet().to(device)
    
    # target.load_state_dict(model.state_dict())
    
    
    actions = [i for i in range(action_num)]
    
    mem = deque(maxlen=mem_size)
    
    frame = 0 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    rs = []
    # print(env._max_episode_steps)
    
    for episode in tqdm(range(episodes)):
        done = False
        s = env.reset()
        
        s = torch.from_numpy(s).unsqueeze(0).float().to(device)
        
        step = 0    
        while done==False:
            frame += 1
            step += 1
            if np.random.rand() < eps:
                sel_a = np.random.choice(actions)
            else:
                sel_a = torch.argmax(model(s)).item()
                
                
            
            s_prime, r, done, _ = env.step(sel_a)
            s_prime = torch.from_numpy(s_prime).unsqueeze(0).float().to(device)
            
            if done:
                if step < 30:
                    r -= 10
                else:
                    r = -1
                    
            if step > 30:
                r += 1
            if step > 100:
                r += 1
            if step > 150:
                r += 1
            if step > 170:
                r += 1
            if step > 190:
                r += 1
            
            
            obs = (s, sel_a, r, s_prime, done)
            
            mem.append(obs)
            s = s_prime
            # print(s)
            
            if frame <= mem_size:
                continue
            
            if done:
                rs.append(step)    
               
            
            batch = random.sample(mem, k=mini_batch_size)
            
            X = torch.empty(mini_batch_size, 4)
            X_prime = torch.empty(mini_batch_size, 4)
            
            for idx, o in enumerate(batch):
                X[idx] = o[0].squeeze()
                X_prime[idx] = o[3].squeeze()
            
            X = X.float().to(device)
            X_prime = X_prime.float().to(device)
            
        
                    
            
            qs = model(X)
            qs_prime = target(X_prime)
            
            # q_new = torch.empty(mini_batch_size).to(device)
            # q_old = torch.empty(mini_batch_size).to(device)
            
            # r_mean = 0
            for idx, o in enumerate(batch):
                o_s, o_a, o_r, o_s_prime, o_done = o 
                o_a = int(o_a)
                # r_mean += o_r 
                # q_old[idx] = qs[idx, o_a]
                if not o_done:
                    q_new = (1-lr) * qs[idx, o_a] + lr * (o_r + gamma*torch.max(qs_prime[idx]))
                    qs[idx, o_a] = q_new
                else:
                    qs[idx, o_a] = o_r 
                    # q_new[idx] = o_r 
            
            # r_mean /= mini_batch_size
            
            
            qs_prime.detach_()
            # q_new.detach_()
            for epoch in range(epochs):
                # qs = model(X)
                # loss = ((qs_prime - qs) ** 2).mean()
                loss = F.smooth_l1_loss(qs, qs_prime)
                
                
                # print(qs)
                # print(qs_prime)
                # print(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print('#########################')
                
            if episode % render_every == 0:
                env.render()
                
        
        
        if end_decay >= episode+1 >= start_decay:
            eps -= decay_amt     
        
        if (episode + 1) % copy_every == 0:
            target.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), 'cart2.pt')
            
        if (episode + 1) % plot_every == 0 and frame > mem_size:
            plt.plot(rs)
            plt.show()
    
    env.close()
    
    
    
    
    
    
    
    
    
    
    
    
    










