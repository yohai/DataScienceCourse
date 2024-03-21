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
2. We put the protein in a solution that has fluorescent molecules, that light up when they attach to the protein. The protein has $B$ binding sites, and the binding probability is $p_f$ when it is folded and $p_u$ when it is unfolded.
3. We measure the protein and count $n$, the number of sites that shining.

Now we can ask all kinds of questions: What will be the distribution of $n$? I found that that the protein has 3 shining sites. What's the probability that it's folded? My microscope was saturated and I'm not sure how many sites are shining, but it's at least 2. What's the probability that it's folded? More challenging: I'm not sure what $B$ is, but I measured this distribution of $n$. What can I say about $N$?

We'll define two _random variables_:
- $X$: whether the protein is folded or not. $X$ takes the values $x_i\in\{0,1\}$. 
- $Y$: the number of shining sites. $Y$ takes the values $y_i\in\{0,1,\dots,B\}$.

The _joint probability_, $p(X=x_i,Y=y_j)$ is the probability of the event $(X=x_i)\cap(Y=y_j)$. Taking the frequentist view, if we repeat the experiment $N$ times, then for large $N$ we expect the number of experiments in which $X=x_i$ AND $Y=y_j$ to be $Np(x_i,y_j)$.

The _marginal probability_ $p(x_i)$ asks about the outcome of the measurement $X$ regardless of the outcome of $Y$. The marginal probability is related to the joint probability by _the sum rule_

$$p(X\!=\!x_i)=\sum_j p(X=x_i, Y=y_j)$$

Calculating the sum in the sum rule is called marginalization.
The _conditional probability_ $p(X=x_i|Y=y_j)$ describes the fraction of times in which you would get the measurement $X=x_i$, out of the experiments in which you found $Y=y_j$. It is _defined_ as

$$
p(X=x|Y=y) = \frac{p(x=X, y=Y)}{p(Y=y)} 
$$ (conditional) 
 
:::{admonition} Active learning
:class: tip
The three $p$ that appear in the above equation are different: one is a conditional probability, another is a joint probability and the third is a marginal one. Make sure you understand which is which.
:::
:::{admonition} Solution
:class: dropdown
$$\mbox{conditional}=\frac{\mbox{joint}}{\mbox{marginal}}$$ 
:::

$X$ and $Y$ are called statistically independent when $p(X|Y)=P(X)$ (that is: $p(x|y)=p(x)$ for every $x$ and $y$).

To ease the notational burden, from now on we'll write $p(x,y), p(x|y)$ to mean $p(X=x,Y=y), p(X=x|Y=y)$ and so on. Eq. {eq}`conditional` can be rephrased as _the product rule_

$$ p(x, y) = p(x|y)p(y) $$

## Expectation, conditional expectation
 **TODO**

## Bayes rule


Together, these rules give rise to the famous Bayes' rule:

$$ 
    p(x|y)=\frac{p(y|x)p(x)}{p(y)} 
$$ (Bayes_rule)

This is a tremendously important equation, don't be deceived by its simplicity. When using Bayes' rule in inference, the three factors on the right hand side are called
- $p(y|x)$ - The _likelihood_
- $p(x)$ - The _prior_
- $p(y)$ - The _evidence_

The term on the left hand side is the _posterior_. Let's see how this works.

+++

## Bayesian inference - first example
Consider the protein from above. There are two random variables of interest: the protein state $F\in\{\text{Folded, Unfolded}\}$ and the number shining sites $N\in\{0,1,\dots, B\}$. 

Say we measured and found $N=8$. What can we say about the state of the protein? Using Bayes' rule, the probability that it's folded is

\begin{align}
    p(\text{Folded}|8)&=\frac{p(8|\text{Folded})p(\text{Folded})}{p(8)}
\end{align}

Bayes' rule simply says that the posterior is the prior, updated by the ratio of the likelihood to the evidence. Let's decompose this. The prior $p(\text{Folded})$ is what we denoted as $q$. It represents our estimation of the folded stater _prior_ to any observation. The  _posterior_ $p(\text{Folded}|8)$ is our updated belief, given the observation $N=8$. 

What is the likelihood $p(8|\text{Folded})$? We know that if the protein is folded then each site is shining with probability $p_f$, independently from the other sites. The probability that $N$ sites are shining are given by the [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)

$$p(n|\text{Folded})=\begin{pmatrix}B\\ n\end{pmatrix}p_f ^n (1-p_f)^{B-n}=\frac{B!}{n!(B-n)!}p_f ^n (1-p_f)^{B-n}$$

We can actually write the joint probability of $N$ and $F$:

$$
    p(n,f)=\frac{B!}{n!(B-n)!}
    \times 
    \begin{cases}
        p_f ^n (1-p_f)^{B-n} q & f=\text{Folded}\\
        p_u ^n (1-p_u)^{B-n} (1-q)& f=\text{Unfolded}\\
    \end{cases}
$$ (joint-protein)

Make sure you understand each term in this equation. Using this, we can calculate both the likelihood

$$
    p(8|\text{Folded}) = \frac{30!}{8!22!}p_f^{8} (1-p_f)^{22} 
$$

and the evidence

$$
p(8) = p(8, \text{Folded})+p(8, \text{Unfolded})
$$

For concreteness, we'll say the protein is folded with probability $q=60\%$, it has $B=20$ binding sites and $p_f=0.2, p_u=0.4$. We can list out all the probabilities:

```{code-cell} ipython3
:tags: [hide-input]

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


B = 20
# ns is an array of integers from 0 to B (including)
ns = np.arange(B + 1)
# scipy.special.comb is the m-choose-n function
# comb is an array of the same size as ns, of the binomial coefficients B!/(n!(B-n)!)
comb = sp.special.comb(B, ns) 

q = 0.5
pf = 0.2
pu = 0.4

def bernoulli(p):
    return comb * p**ns * (1-p)**(B-ns)

# joint[f,n] is the joint probability of n shining sites
# in the folded/unfolded state f (0 is folded, 1 unfolded)  
joint = np.array([bernoulli(pf)*q, bernoulli(pu)*(1-q)])

plt.plot(joint[0], 'o-', label='p(n, folded)')
plt.plot(joint[1], 'o-', label='p(n, unfolded)')
plt.gca().set_xticks(np.arange(0,20,2))
plt.axvline(8, ls='--', c='r')
plt.legend();
```

Our posterior belief about the state of the protein are

```{code-cell}
N = 8
p_folded_posterior = joint[0, N+1]/(joint[0, N+1] + joint[1, N+1])
p_unfolded_posterior = joint[1, N+1]/(joint[0, N+1] + joint[1, N+1])
print(f'p(folded | n=8) = {p_folded_posterior:.2f}')
print(f'p(unfolded | n=8) = {p_unfolded_posterior:.2f}')
```

and analytically:

\begin{align}
    p(\text{Folded}|12) & = \frac{\frac{12!}{8!12!}p_f^{12} (1-p_f)^{8} \times q}{p(12)} \\
    p(\text{Unfolded}|12) & = \frac{\frac{12!}{8!12!}p_u^{12} (1-p_u)^{8} \times (1-q)}{p(12)} \\
\end{align}

Note that this ratio of 96:4 is exactly the ratio of the orange and blue curves in the figure, evaluated at $N=12$. We can plot our estimation probability as a function of $N$:

```{code-cell}
p_folded_posterior_list = joint[0, :]/(joint[0, :] + joint[1, :])
p_unfolded_posterior_list = joint[1, :]/(joint[0, :] + joint[1, :])
plt.plot(ns, p_folded_posterior_list, 'o-', label='p(folded | n)')
plt.plot(ns, p_unfolded_posterior_list, 'o-', label='p(unfolded | n)')
plt.legend();
```


:::{admonition} Active learning
:class: tip
Assume that your detector is not so great (or you're just a bad experimentalist) and you can't get exact readings like $N=12$. Instead, the detector tells you that $8\le N \le 13$. What is now your estimation of the probability that the protein is folded?
:::

+++

# Information

We defined statistical as the condition where $p(x|y)=p(x)$ for every $x$ and $y$. This means that the distribution of $X$ is not changed if we condition it on any particular value of $y$. Put differently: knowledge about the the random variable $Y$ does not add any information about $X$. What does _information_ mean, quantitatively, in this context? 

Information is a fundamental and elusive concept which comes up in probability, data science, statistical mechanics and a bunch of other fields. You probably know that the Shannon Entropy, or Shannon Information, of a random variable $X$ is defined as

$$
H(X) = \sum_i - p(x_i) \log p(x_i) = \avg{ \log \left(\frac{1}{p(X)}\right) }
$$ (entropy)

:::{warning}
Here I'm using logarithms in base 2, because it's convenient when talking about information. In other context it makes more sense to use the natural logarithm, in which case the units are called _nats_ and not bits. It almost never matters, and I'll use "log" pretty freely without specifying which basis is used. 
:::

## Information in binary variables

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

## Conditional information
Let's take another look at the [simple protein example](protein-example) from before. Before measuring $N$, the information content of $F$ (i.e. the amount of information that we need in order to determine $F$) is simply $H_2(q)$, For $q=0.6$ this equals 0.97 bits.

What's the information after we measured, $N=8$ as before? We are now $96\%$ certian that the protein is unfolded and so the information content of $F$ _conditioned_ on $N=8$ is

$$
    H(F|12) = \avg{\log\frac{1}{p(F|12)}} = H_2(0.96) = 0.24 \text{ bits}
$$

We see that the measurement $N=8$ provided us with $\approx 0.73$ bits. 

The quantity $H(X|y)$ is the called the conditional information of $X$ given that $Y=y$. It has very intuitive properties:

- If $X$ and $Y$ are independent then $H(X|y)=H(x)$ for every $y$.
- $H(X|y) \le H(X)$. That is, knowing the value of $Y$ cannot possibly make us know _less_ about $X$.

**TODO**: Look at Bishop, including cure-fitting re-visited and adopt some of the structure. 

```{code-cell}

```

::: {admonition} Reference
::::{bibliography}
:style: unsrt
::::
:::
