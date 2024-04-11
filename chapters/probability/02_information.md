---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
---

# Information

We defined statistical independence as the condition where $p(x|y)=p(x)$ for every $x$ and $y$. This means that the distribution of $X$ is not changed if we condition it on any particular value of $y$. Put differently: knowledge about the the random variable $Y$ does not add any information about $X$. What does _information_ mean, quantitatively, in this context? 

Information is a fundamental and elusive concept which comes up in probability, data science, statistical mechanics and a bunch of other fields. You probably know that the Shannon Entropy, or Shannon Information, of a random variable $X$ is defined as

$$
H(X) = \sum_i - p(x_i) \log p(x_i) = \avg{ \log \left(\frac{1}{p(X)}\right) }
$$ (entropy)

:::{warning}
Here I'm using logarithms in base 2, because it's convenient when talking about information. In other context it makes more sense to use the natural logarithm, in which case the units are called _nats_ and not bits. It almost never matters, and I'll use "log" pretty freely without specifying which basis is used. 
:::

## Example: Information in binary variables

Let's try to get some intuition for why it makes sense to interpret this as Information. First, we see that $H$ is the average of $\log\frac{1}{p(x)}$. We can think of $\log \frac{1}{p(x_i)}$ as the amount of information of the observation $x_i$. That is, the information we'd gain if we learned that $X=x_i$. It is sometimes called the "surprise". Imagine you're playing [Guess Who](https://en.wikipedia.org/wiki/Guess_Who%3F) and you have 30 suspects, of which:
- 13 are women
- 5 wear glasses
- 9 have curly hair
- 1 is named Jack
- etc ...

The goal of the game is to find who the suspect is, and you can ask yes/no questions. What would be a good question to ask (=what would be a good observable to measure)? 

We think of the chosen character as a random variable $X\in \{\text{Jack, Tiffany, Bill...}\}$. We can also define the random variables for such as $G\in\{\text{True, False}\}$ which denote whether the character wars Glasses or is a Woman, respectively. If we have no prior beliefs about who the suspect is, we assume that the distribution of $X$ is uniform across all the names, $p(x_i)=\frac{1}{30}$. According to Eq. {eq}`entropy`, the information content of $X$ is thus $H(X)=\avg{\log 30} =\log 30$.

We can ask "is your character Jack?", i.e. measure the observable $J = X \text{ is Jack}$. The answer will almost certainly be "no". This answer will not surprise us and we'll gain very little information, only $\log \frac{30}{29}\approx 0.05$ bits, from the observation $X\ne\text{Jack}$. However, if the answer is "yes", which happens with probability 1/30, we'd narrow it down to a single option, and win the game. Note that the information that we'd gain is $\log 30$, which is exactly the total information content of $X$ - there's nothing more to learn. On average, we expect to gain 

$$H(J) = \frac{1}{30}\log 30+\frac{29}{30}\log\frac{30}{29}=0.15 \text{ bits.} $$

In contrast, if we measure the observable $W = X \text{ is a woman}$ we're certain that whatever the answer would be, we'd still have between 13 and 17 options left. That is, for both possible answers, we'd gain a decent amount of information, but definitely not all of it. How much?

$$ H(W) = \frac{13}{30}\log \frac{30}{13}+\frac{17}{30}\log\frac{30}{17} = 0.68 \text{ bits.} $$

What would be the best yes/no question to ask? The one that gives us the most information? If we ask a question (=measure a binary observable) whose answer is yes with probability $p$ then the information we'd gain, on average, is
  
$$H_2(p)=-\Big(p\log p +(1-p)\log(1-p)\Big)$$

You have probably seen this equation before, and it was called "mixing entropy" or something of that sort. It is maximal for $p=\frac{1}{2}$, i.e. the best yes/no question we can ask is one that has 50% chances of being answered "yes". In other words: the one that we are least certain about its result.

## Conditional, joint, mutual information
Let's take another look at the [simple protein example](protein-example) from before. Before measuring $N$, the information content of $S$ (i.e. the amount of information that we need in order to determine $F$) is simply $H_2(q)$, For $q=0.6$ this equals 0.97 bits.

What's the information after we measured $N=5$ as before? We are now $76\%$ certian that the protein is unfolded and so the information content of $F$ _conditioned_ on $N=5$ is

$$
    H(S|12) = \avg{\log\frac{1}{p(F|12)}} = H_2(0.76) = 0.82 \text{ bits}
$$

We see that the measurement $N=5$ provided us with $\approx 0.15$ bits of information about $S$. The quantity 

$$
    H(X|y)=\avg{\log\frac{1}{p(x|y)}}_X
$$ 

is the called the conditional information of $X$ given that $Y=y$. Note that the averaging is performed over $X$. Here is the conditional information of $S$ for every possible measurement of $N$:


```{code-cell} ipython3
:tags: [hide-input]
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
plt.plot(ns, pf_post, '-o', label=r'$p(\mbox{S}\vert N)$')
plt.plot(ns, h_conditional, '-o', label=r'$H(\mbox{S}\vert N)$')
plt.legend(fontsize=16)
plt.gca().set_xlabel('N')
plt.show()
```

You can see that, for example, if we measure $N=10$ then there is almost no uncertainty left in $S$. We will be pretty sure (in fact, 99.75% certain) that the protein is unfolded. However, this is a very unlikely measurement, only $p(10)=p(10|\text{F})+p(10|\text{U})=0.01$% of measurements will be $N=10$. 

The amount of information left in $S$ on average is called the _conditional information_.

$$
    H(X|Y) = \avg{H(X|y)}_y = \avg{\log \frac{1}{p(x|y)}}_{X, Y}
$$

It has a few intuitive properties. First, one can show that $H(x|y)=H(x)$ for every $x,y$ if and only if $X$ and $Y$ are independent (prove!). Second, applying [the product rule](product_rule) immediately shows a simple relation between the joint information $H(X,Y)$ and the conditional one

$$
    H(X, Y) = - \avg{\log p(X, Y)}= - \avg{\log p(X|Y) + \log p(Y)} = H(X | Y)+ H(Y)
$$

Intuitively, we interpret this as the statement that the information content of $X,Y$ jointly is the information that you learn when you measure $Y$ plus the information you learn when you measure $X$ after you already know $Y$. Since this is symmetric, it also works the other way around: 

$$
    H(X, Y) = H(X | Y)+ H(Y) = H(Y | X)+ H(X)
$$

It is nice to see that these definitions capture a very basic truth about information:

$$
\begin{align}
    H(X|Y) &= -\avg{\log p(X|Y)}_{X,Y}= -\avg{\avg{\log p(X|Y)}_Y \vphantom{\frac12}}_{X} \\
    &\le -\avg{\log \left(\avg{p(X|Y)}_Y \vphantom{\int}\right)}_X = -\avg{\log p(X)}_X = H(X)
\end{align}
$$ (conditional_information)

In the transition we made use of the fact that $\avg{\log X}\ge\log\avg{X}$ for any random variable $X$, due to the convexity of the logarithm function (a corollary of  [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)). The interpretation of this result, $H(X|Y)\le H(X)$, is that knowing the value of $Y$ cannot possibly, on average, make us know _less_ about $X$.


```{admonition} Active learning
:class: tip
We found that $H(X|Y)\le H(X)$ when you average over $Y$. Can it be that for a specific observation $y$ we'd have $H(X,y)>H(X)$, i.e. that there's _more_ uncertainty about $X$ after measuring $Y=y$?
```

A natural quantity to define is the _mutual information_, defined as

$$
    M(X,Y) = H(X|Y)-H(X) = H(Y|X)-H(Y)=H(X,Y)-H(X)-H(Y)
$$

It is the amount of information shared between $X$ and $Y$. Eq. [](conditional_information) tells us that is always non-negative $M(X,Y)\ge 0$.   TODO: add Ven Diagram.


 
```{admonition} Active learning
:class: tip
Say you have a very bad detector that only gives you a binary reading: it says whether $N\le k$ or $N>k$ for some $k$. Fortunately, you can turn the knobs to select $k$. What would be the best value to select in order to best distinguish between a folded and an unfolded protein?

Hint: In other words, what $k$ maximizes the mutual information between the observed quantity and the protein state
```
