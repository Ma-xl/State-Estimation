#!/usr/bin/env python
# coding: utf-8

# Q4

# In[1]:


import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Define basic values.
K = 12709
covq = 2.26e-5
covr = 3.67e-4


# In[2]:


# Define matrices.
A = np.tri(K,K)
C = np.diag(np.full(K,1))
H = np.vstack((np.linalg.inv(A),C))
Q = np.diag(np.full(K,covq))
R = np.diag(np.full(K,covr))
W = scipy.linalg.block_diag(Q,R)


# In[3]:


# Calculation of the left-hand side.
lhs_1 = np.matmul(np.transpose(H),np.linalg.inv(W))
lhs = np.matmul(lhs_1,H)
lhs


# In[4]:


# Plot the sparsity pattern.
plt.spy(lhs)


# Q5

# In[5]:


# Load MATLAB data.
data = scipy.io.loadmat('dataset1.mat')


# In[6]:


# Define algorithm.
def rob_pos(delta):
    # Define data variables and coefficients.
    x_true = data['x_true']
    data_size = x_true.shape[0]
    A = 1
    C = 1
    Q = 2.26e-5
    R = 3.67e-4
    speed = data['v']
    x_c = data['l']
    r = data['r']
    t = data['t']
    f = 10
    y = x_c - r
    v = speed/f
    
    # Create empty array.
    
        # Forward.
    P_ch_f = np.zeros(data_size)
    x_ch_f = np.zeros(data_size)
    K = np.zeros(data_size)
    P_h_f = np.zeros(data_size)
    x_h_f = np.zeros(data_size)
    
        # Backward.
    x_h = np.zeros(data_size)
    P_h = np.zeros(data_size)
    
    # Initial conditions for forward.
    P_ch_f[0] = R
    x_ch_f[0] = y[0]
    K[0] = P_ch_f[0]*C*(C*P_ch_f[0]*C+R)**(-1)
    P_h_f[0] = (1-K[0]*C)*P_ch_f[0]
    x_h_f[0] = x_ch_f[0]+K[0]*(y[0]-C*x_ch_f[0])
    
    # Forward algorithm.
    for i in range(1, data_size):
        #Calculate only on certain timestep.
        if i%delta != 0:
            C = 0
        else:
            C = 1
            
        P_ch_f[i] = A*P_h_f[i-1]*A+Q
        x_ch_f[i] = A*x_h_f[i-1]+v[i]
        K[i] = P_ch_f[i]*C*(C*P_ch_f[i]*C+R)**(-1)
        P_h_f[i] = (1-K[0]*C)*P_ch_f[i]
        x_h_f[i] = x_ch_f[i]+K[i]*(y[i]-C*x_ch_f[i])
    
    # Initial conditions for backward.
    x_h[data_size-1] = x_h_f[data_size-1]
    P_h[data_size-1] = P_h_f[data_size-1]
    
    # Backward algorithm.
    for i in range(data_size-1, 0, -1):
        x_h[i-1] = x_h_f[i-1]+(P_h_f[i-1]*A*P_ch_f[i]**(-1))*(x_h[i]-x_ch_f[i])
        P_h[i-1] = P_h_f[i-1]+(P_h_f[i-1]*A*P_ch_f[i]**(-1))*(P_h[i]-P_ch_f[i])*(P_h_f[i-1]*A*P_ch_f[i]**(-1))
    
    # Calculating error.
    error_size = x_h.shape[0]
    error = np.zeros(error_size)
    for i in range(0, error_size):
        error[i] = x_h[i] - x_true[i]
    
    # Calculating uncertainty envelope.
    var = P_h
    unc_env = 3*np.sqrt(var)
    
    # Plots.
    fig, ax = plt.subplots(2, figsize=(7,7))
    ax[0].plot(t, error, label = 'Error', linewidth = 0.6)
    ax[0].fill_between(np.squeeze(t), -unc_env, unc_env, label = 'Uncertainty envelope', linestyle = ':', color = 'yellow')
    ax[0].legend()
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Error (m)')
    ax[1].hist(error, bins = 40, rwidth = 0.9)
    ax[1].set_xlabel('Error (m)')
    ax[1].set_ylabel('Quantity')


# In[7]:


# Plots for delta=1.
rob_pos(1)


# In[8]:


# Plots for delta=10.
rob_pos(10)


# In[9]:


# Plots for delta=100.
rob_pos(100)


# In[10]:


# Plots for delta=1000.
rob_pos(1000)


# In[ ]:




