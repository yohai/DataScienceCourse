#%%

import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True
K = 12
# ns is an array of integers from 0 to B (including)
ns = np.arange(K + 1)

# comb is an array of the same size as ns, 
# of the binomial coefficients B!/(n!(B-n)!)
comb = sp.special.comb(K, ns) # this is the m-choose-n function

q = 0.6
pf = 0.2
pu = 0.4

def bernoulli(p):
    return comb * p**ns * (1-p)**(K-ns)

# joint[f,n] is the joint probability of n shining sites
# in the folded/unfolded state f (0 is folded, 1 unfolded)  
joint = np.array([bernoulli(pf)*q, bernoulli(pu)*(1-q)])

pf_post = joint[0]/(joint[0]+joint[1])
h_conditional = - pf_post * np.log2(pf_post) - (1 - pf_post) * np.log2(1-pf_post)
plt.plot(ns, pf_post, '-o', label=r'$p(\mbox{F}\vert N)$')
plt.plot(ns, h_conditional, '-o', label=r'$H(\mbox{F}\vert N)$')
plt.legend(fontsize=16)
plt.gca().set_xlabel('N')
# %%
joint[:, 10].sum()
# %%
