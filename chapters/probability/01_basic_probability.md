---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
---

# Basic probability concepts
```{admonition} Best references 
:class: note
I highly recommend Mackay's book {cite}`mackay` which is [freely available online](http://www.inference.org.uk/itprnn/book.pdf), and the first chapter of Murphy {cite}`murphy` or Bishop {cite}`bishop`.
```

The mathematical theory of probability, which you should know at least in part, is formulated in terms of _sample space_, _events_, _random variables_, and _probabilities_. Given an event $A$ the probability $p(A)$ is a number between 0 and 1 that quantifies the probability that $A$ is true. 

What does that mean? There are two ways to look at it:

* **Frequentist interpretation** If we repeat the same "experiment" $N$ times, then $A$ will happen  $N\cdot p(A)$ times when $N$ is very large. This is the essence of the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers).

* **Bayesian interpretation** In the Bayesian view, probabilities represent *uncertainties*. The statement $p(A)=0.73$ means that we are 73% certain that the event $A$ is true.

The frequentist interpretation is more easily quantifiable, and is suitable in many cases where an experiment is (effectively) repeated many times. It may come up while answering these kind of questions:
* I have 1Kg atom of Plutomium-241 (yay!). How much of it will decay in the next 24 hours?
* .... **TODO** ....

The Bayesian interpretation is harder to quantify, as "beliefs" is quite a vague term. While it can be formulated axiomatically, e.g. [Cox's theorem](https://en.wikipedia.org/wiki/Cox%27s_theorem), we will not deal with this formally, and rely on intuition (we're still physicists...). The Bayesian interpretation comes up when we want to answer questions of a different flavour, such as:
* What's the probability that Mr. X is the murderer in this criminal case?
* Is this email spam or legit?
* I did this experiment and got this reading on my sensor. What can I say about my system given the measurement? How certain am I in my statements?

In these kinds of situations it makes no sense to think about "repeating the experiment" many times. The email is either legit or spam. Mr. X either killed the poor man or not. There's a single true answer to the question "what is the structure of my system?". We just _don't know_ it.  Laymen use "probabilities" to describe both kind of situations, but they're quite distinct in their foundation. 

In this course we will use "probabilities" in the Bayesian sense, which is more suitable for _inference_ problems, the typical situation in physics research. It happens often that we have some measurements, information, prior knowledge, or a combination of them, and we wish to combine them to _quantitatively infer_ something about it. The quantitative inference is the heart of this course. 

```{image} https://imgs.xkcd.com/comics/frequentists_vs_bayesians.png
:alt: XKCD panel on Bayesian
:align: center
```

+++



+++

(protein-example)=
## Joint, marginal, conditional probabilities
Consider the following "physical" scenario:
1. A protein can be in either of two states, say: folded and unfolded. The probability of it being folded is $q$.
2. We put the protein in a solution that has fluorescent molecules, that light up when they attach to the protein. The protein has $K$ binding sites, and the binding probability is $p_f$ when it is folded and $p_u$ when it is unfolded.
3. We measure the protein and count $N$, the number of sites that are shining.

Now we can ask all kinds of questions: 

- What will be the distribution of $N$? 
- I found that that the protein has 3 shining sites. What's the probability that it's folded? My microscop
- e was saturated and I'm not sure how many sites are shining, but it's at least 2. What's the probability that it's folded? 
- More challenging: I'm not sure what $K$ is, but I measured this distribution of $n$. What can I say about $N$?

These are _inference_ questions. We want to infer something about about one variable given some information about the other. 
We'll define two _random variables_:
- $S$: The state of hte protein. $F$ takes values in $\{\text{F},\text{U}\}$. 
- $N$: the number of shining sites. It takes values in $\{0,1,\dots,K\}$.


Our first step is to consider all possible outcomes, or in other words,  all possible combinations of $S$ and $N$ and consider their _joint probability_ $p(n, s)$. This describes the probability that both $N=n$ and $S=s$ at the same time. 

How do model our knowledge of the binding probability? If the protein is folded/unfolded then each site is shining with probability $p_f$/$p_u$, independently from the other sites. In either state, the probability that $N$ sites are shining is given by the [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution), whose density function is

$$
    p_B(N; p, K) = \begin{pmatrix}K\\ n\end{pmatrix}p ^n (1-p)^{K-n}=\frac{B!}{n!(K-n)!}p ^n (1-p)^{K-n}
$$ (bimonial_distribution)

where $p$ is either $p_u$ or $p_f$. If we know $S$, then Eq. [](bimonial_distribution) tells us what is the distribution of $N$. This is called _conditional probabiltiy_ and is denoted by $p(x|y)$ (probability of $x$ given $y$). That is,

$$
\begin{split}
p(N|S=\text{F}) &= p_B(N; p_f, S) \\
p(N|S=\text{S}) &= p_B(N; p_u, S) 
\end{split}
$$ (protein_conditional)

The conditional probability is related to the joint probability by the _product rule_,

$$ 
    p(x, y) = p(x|y)p(y) =p(y|x)p(x)
$$ (product_rule)

Combining Eq. [](product_rule) with Eq. [](protein_conditional) gives us explicitly the joint probability

$$
    p(N, F) = p(N|F)p(F) 
    = \begin{cases}
        p_B(N ; p_f, K) q & S=\text{F} \\ 
         p_B(N ; p_u, K) (1-q) & S=\text{U}
    \end{cases}
$$


The _marginal probability_ $p(x_i)$ asks about the outcome of the measurement $X$ regardless of the outcome of $Y$. The marginal probability is related to the joint probability by _the sum rule_

$$p(X\!=\!x_i)=\sum_j p(X=x_i, Y=y_j)$$

Calculating the sum is called marginalization.


```{admonition} Active learning
:class: tip
The three $p$ that appear in Eq. [](product_rule) are different: one is a conditional probability, another is a joint probability and the third is a marginal one. Make sure you understand which is which.
```
```{admonition} Solution
:class: dropdown
$$
    \text{joint}=\text{conditional}\times\text{marginal}
$$ 
```

For concreteness, we'll say the protein is folded with probability $q=60\%$, it has $B=12$ binding sites and $p_f=0.2, p_u=0.4$. Here's how it looks like:

```{code-cell} ipython3
:tags: [hide-input]

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(10, 2))
# Display the data
cax = ax.matshow(joint, cmap='Reds')

# Add text annotations
for (i, j), val in np.ndenumerate(joint):
    if val < 0.01:
        cell = '<0.01'
    else:
        cell = f'{val:.2f}'

    ax.text(j, i, cell, ha='center', va='center', fontsize=10, 
            c='k' if val < 0.07 else 'w')

ax.set_xticks(range(K+1))
ax.set_xticklabels(range(K+1))
ax.set_yticks([0, 1])
ax.set_yticklabels(['F', 'U'])
plt.show()
```


$X$ and $Y$ are called statistically independent when $p(X|Y)=P(X)$ (that is: $p(x|y)=p(x)$ for every $x$ and $y$). To ease the notational burden, from now on we'll write $p(x,y), p(x|y)$ to mean $p(X=x,Y=y), p(X=x|Y=y)$ and so on.

:::{admonition} Active learning
:class: tip
Show that if $X$ and $Y$ are statistically independent then the joint probability matrix if of rank 1.
:::

## Expectation, conditional expectation
 **TODO**

## Bayes rule


The product rule, Eq. [](product_rule), can be rephrased as the famous Bayes' theorem,

$$ 
    p(x|y)=\frac{p(y|x)p(x)}{p(y)} 
$$ (bayes_rule) 

This is a tremendously important equation, don't be deceived by its simplicity. When using Bayes' rule in inference, the three factors on the right hand side are called
- $p(y|x)$ - The _likelihood_
- $p(x)$ - The _prior_
- $p(y)$ - The _evidence_

The term on the left hand side is the _posterior_. Let's see how this works.

## Bayesian inference - first example
Consider the protein from above.


Now we can answer the question. Say we measured $N=5$. Using Bayes' rule, the probability that the protein is folded is

$$
    p(\text{F}|5)=\frac{p(5|\text{F})p(\text{F})}{p(5)}
$$

Bayes' rule simply says that the posterior is the prior, updated by the ratio of the likelihood to the evidence. Let's decompose this. The prior $p(\text{F})$ is what we denoted as $q$. It represents our estimation of the folded stater _prior_ to any observation. The  _posterior_ $p(\text{F}|5)$ is our updated belief, given the observation $N=5$. 

The likelihood $p(5|\text{F})$ is given by Eq. [](protein_conditional), and the evidence is

$$
p(5) = p(5, \text{F})+p(5, \text{U})
$$


Our posterior belief about the state of the protein are

```{code-cell}
N = 5
p_folded_posterior = joint[0, N]/(joint[0, N] + joint[1, N])
p_unfolded_posterior = joint[1, N]/(joint[0, N] + joint[1, N])
print(f'p(folded | n=5) = {p_folded_posterior:.2f}')
print(f'p(unfolded | n=5) = {p_unfolded_posterior:.2f}')
```

Note that this ratio of 74:26 is exactly the ratio 0.09:0.03 between the top and bottom cells in the 5-th column in the joint probability matrix. We can plot our estimation probability as a function of $N$:

```{code-cell}
p_folded_posterior_list = joint[0, :]/(joint[0, :] + joint[1, :])
p_unfolded_posterior_list = joint[1, :]/(joint[0, :] + joint[1, :])
plt.plot(ns, p_folded_posterior_list, 'o-', label='p(folded | n)')
plt.plot(ns, p_unfolded_posterior_list, 'o-', label='p(unfolded | n)')
plt.legend();
```


:::{admonition} Active learning
:class: tip
Assume that your detector is not so great and you can't get a precise reading of $N$. Instead, you can only assert that $4\le N \le 6$. What is now your estimation of the probability that the protein is folded?
:::

+++

**TODO**: Look at Bishop, including cure-fitting re-visited and adopt some of the structure. 



::: {admonition} Reference
::::{bibliography}
:style: unsrt
::::
:::
