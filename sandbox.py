#%%
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt

B = 20
ns = np.arange(B + 1)
comb = sc.special.comb(B, ns)

q = 0.5
pf = 0.2
pu = 0.4

def bernoulli(p):
    return comb * p**ns * (1-p)**(B-ns)
plt.plot(bernoulli(pf))
plt.plot(bernoulli(pu))
# %%
c1 = [2, 7]
c2 = [6, 11]

def probs(p, c):
    b = bernoulli(p)
    return np.concatenate([
        [b[:c[0]].sum()], 
        b[c[0]:c[1]+1],
        [b[c[1]+1].sum()], 
    ])

p1 = probs()


# %%
ns[c1[1]:]

# %%
