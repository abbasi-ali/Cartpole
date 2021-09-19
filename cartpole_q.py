# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:13:32 2021

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
import cv2 
from PIL import Image 


# class CartNet(nn.Module):
#     def __init__(self, h, w, outputs):
#         super(CartNet, self).__init__()
#         self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=3)
#         # self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
#         # self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         # self.bn3 = nn.BatchNorm2d(32)

#         # Number of Linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so compute it.
#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
        
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 5, 3), 4, 2), 3, 1)
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 5, 3), 4, 2), 3, 1)
#         linear_input_size = convw * convh * 64
        
        
#         self.lin1 = nn.Linear(linear_input_size, 512)
#         self.lin2 = nn.Linear(512, 256)
#         self.lin3 = nn.Linear(256, 64)
#         self.lin4 = nn.Linear(64, outputs)

#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = x.to(device)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.lin1(x.view(x.size(0), -1)))
#         x = F.relu(self.lin2(x))
#         x = F.relu(self.lin3(x))
#         x = self.lin4(x)
        
#         return x 

HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32 
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
class CartNet(nn.Module):

    def __init__(self, h, w, outputs):
        super(CartNet, self).__init__()
        self.conv1 = nn.Conv2d(4, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE) 
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * HIDDEN_LAYER_3
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))        
    
    
    
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

env = gym.make('CartPole-v1')

# high = torch.from_numpy(env.high).to(device)
# low = torch.from_numpy(env.low).to(device)


# chunk_p = 20 
# chunk_v = 20 

action_num = 2
episodes = 10_000

lr = 1
gamma = 0.8
eps = 1.0 

min_eps = 0.001
start_decay = 1
end_decay = 100
decay_exp = 0.0005
decay_amt = (eps - min_eps) / (end_decay - start_decay)


# render_every = 10 
copy_every = 5 
plot_every = 5
save_every = 5

mini_batch_size = 16
mem_size = 1_000

epochs = 1

in_channels = 4 
in_h = 250 
in_w = 400 


model = CartNet(in_h, in_w, 2).to(device)
target = CartNet(in_h, in_w, 2).to(device)

# target.load_state_dict(model.state_dict())


actions = [i for i in range(action_num)]

mem = deque(maxlen=mem_size)

frame = 0 

optimizer = optim.Adam(model.parameters(), lr=0.0005)

rs = []
print(env._max_episode_steps)

model.train()
target.eval()


for episode in tqdm(range(episodes)):
    done = False
    env.reset()
    
    sts = np.zeros((in_h, in_w, in_channels))
    
    for cnt in range(in_channels):
        s = np.ascontiguousarray(env.render(mode='rgb_array'))
        s = s[400 -  in_h:, :, :]
        s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
        s[s < 255] = 0
        s = cv2.resize(s, (in_w, in_h))
        sts[:, :, cnt] = s
        
    sts = torch.from_numpy(sts).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255   
    # print(s.shape)
    # s = get_wind(s, None, None)
    
    step = 0    
    sts2 = sts.clone()
    while done==False:
        frame += 1
        step += 1
        
        # if eps > min_eps:
        #     eps *= (1 - decay_exp)
        
        if np.random.rand() < eps:
            sel_a = np.random.choice(actions)
        else:
            sel_a = torch.argmax(model(sts)).item()
            
        
        _, r, done, _ = env.step(sel_a)
        
        
        # if done:
        #     if step < 30:
        #         r -= 10
        #     else:
        #         r = -1
                    
        # if step > 30:
        #     r += 1
        # if step > 100:
        #     r += 1
        # if step > 150:
        #     r += 1
        # if step > 170:
        #     r += 1
        # if step > 190:
        #     r += 1
        
        
        if not done or step == 500:
            r = step 
        else:
            r = -100
               
    
        if done:
            rs.append(step)    
        
        
        s_prime = np.ascontiguousarray(env.render(mode='rgb_array'))
        s_prime = s_prime[400 -  in_h:, :, :]
        s_prime = cv2.cvtColor(s_prime, cv2.COLOR_RGB2GRAY)
        s_prime[s_prime < 255] = 0
        s_prime = cv2.resize(s_prime, (in_w, in_h))
        s_prime = torch.from_numpy(s_prime).float().to(device) / 255 
        # if frame == 50:
        #     for i in range(4):
        #         tmp = sts[0, i].unsqueeze(0)
        #         # print(tmp.shape)
        #         tmp = tmp.detach().cpu().numpy().transpose((1, 2, 0))
        #         tmp = np.repeat(tmp, 3, 2) * 255 
        #         tmp = tmp.astype(np.uint8)
        #         # print(tmp.shape)
        #         Image.fromarray(tmp).save('F{}-{}.jpg'.format(frame, i))
                
        
        sts2 = torch.roll(sts, shifts=-1, dims=1)
        sts2[0, -1] = s_prime
        # if done:
        #     for i in range(4):
        #         tmp = sts2[0, i].unsqueeze(0)
        #         # print(tmp.shape)
        #         tmp = tmp.detach().cpu().numpy().transpose((1, 2, 0))
        #         tmp = np.repeat(tmp, 3, 2) * 255 
        #         tmp = tmp.astype(np.uint8)
        #         # print(tmp.shape)
        #         Image.fromarray(tmp).save('F2{}-{}.jpg'.format(frame, i))
        
        
        
                
                
                
                
        
        # print(sts2.shape)
        
        # s_prime = torch.from_numpy(s_prime).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255     
        
        
        # s_prime = get_wind(s_prime)
        # print(s_prime.shape)
        obs = (sts, sel_a, r, sts2, done)
        
        mem.append(obs)
        sts = sts2.clone()
        
        if frame <= mem_size:
            continue
        
        batch = random.sample(mem, k=mini_batch_size)
        
        X = torch.empty(mini_batch_size, in_channels, in_h, in_w)
        X_prime = torch.empty(mini_batch_size, in_channels, in_h, in_w)
        
        for idx, o in enumerate(batch):
            X[idx] = o[0].squeeze()
            X_prime[idx] = o[3].squeeze()
        
        X = X.float().to(device)
        X_prime = X_prime.float().to(device)
        
    
                
        
        qs = model(X)
        qs_prime = model(X_prime)
        
        # r_mean = 0
        for idx, o in enumerate(batch):
            o_s, o_a, o_r, o_s_prime, o_done = o 
            o_a = int(o_a)
            # r_mean += o_r 
            
            if not o_done:
                # q_new = (1-lr) * qs[idx, o_a] + lr * (o_r + gamma*torch.max(qs_prime[idx]))
                q_new = o_r + gamma*torch.max(qs_prime[idx])
                qs_prime[idx, o_a] = q_new
            else:
                qs_prime[idx, o_a] = o_r 
                
        
        # r_mean /= mini_batch_size
        
        
        qs_prime.detach_()
        
        for epoch in range(epochs):
            # qs = model(X)
            # loss = ((qs_prime - qs) ** 2).mean()
            
            loss = F.smooth_l1_loss(qs, qs_prime)
            # print(loss.item())
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    
    
    if end_decay >= episode+1 >= start_decay:
        eps -= decay_amt    
    
    if (episode + 1) % copy_every == 0:
        target.load_state_dict(model.state_dict())
        # torch.save(model.state_dict(), 'cart_conv.pt')
        
    if (episode + 1) % save_every == 0:
        torch.save(model.state_dict(), 'cart_conv.pt')
        
    if (episode + 1) % plot_every == 0:
        plt.plot(rs)
        plt.show()

env.close()























