# Refresher on Probability


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

## Basic definitions
Consider the following "physical" scenario:
1. A protein can be in either of two states, say: folded and unfolded. The probability of it being folded is $q$.
2. We put the protein in a solution that has fluorescent molecules, that light up when they attach to the protein. The protein has $B$ binding sites, and the binding probability is $p_f$ when it is folded and $p_{uf}$ when it is unfolded.
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

Together, these rules give rise to the famous Bayes' rule:

$$ 
    p(x|y)=\frac{p(y|x)p(x)}{p(y)} 
$$ (Bayes_rule)

This is a tremendously important equation, don't be deceived by its simplicity. When using Bayes' rule in inference, the three factors on the right hand side are called
- $p(y|x)$ - The _likelihood_
- $p(x)$ - The _prior_
- $p(y)$ - The _evidence_

The term on the left hand side is the _posterior_. We'll get back to these concepts later.

:::{admonition} Example
:class: tip

Medical examination, some other 
:::

## Information

We defined statistical as the condition where $p(x|y)=p(x)$ for every $x$ and $y$. This means that the distribution of $X$ is not changed if we condition it on any particular value of $y$. Put differently: knowledge about the the random variable $Y$ does not add any information about $X$. What does _information_ mean, quantitatively, in this context? 

Information is a fundamental and elusive concept which comes up in probability, data science, statistical mechanics and a bunch of other fields. You probably know that the Shannon Entropy, or Shannon Information, of a random variable $X$ is defined as

$$
H(X) = \sum_i - p(x_i) \log p(x_i) = \avg{ \log \left(\frac{1}{p(X)}\right) }
$$ (entropy)

:::{warning}
Here I'm using logarithms in base 2, because it's convenient when talking about information. In other context it makes more sense to use the natural logarithm. It almost never matters, and I'll use "log" pretty freely without specifying which basis is used. 
:::

Let's try to get some intuition for why it makes sense to interpret this as Information. First, we see that $H$ is the average of $\log\frac{1}{p(x)}$. We can think of $\frac{1}{p(x_i)}$ as the amount of information we'd gain if learned that $X=x_i$. It is sometimes called the "surprise". Imagine you're playing [Guess Who](https://en.wikipedia.org/wiki/Guess_Who%3F) and you have 30 suspects, of which:
- 15 are women
- 5 wear glasses
- 9 have curly hair
- 1 is named Jack
- etc ...

The goal of the game is to find who the suspect is, and you can ask yes/no questions. What would be a good question to ask (=what would be a good observable to measure)? 

We think of the chosen character as a random variable $X\in \{\text{Jack, Tiffany, Bill...}\}$. We can also define the random variables for such as $G\in\{\text{True, False}\}$ which denote whether the character wars Glasses or is a Woman, respectively. If we have no prior beliefs about who the suspect is, we assume that the distribution of $X$ is uniform across all the names, $p(x_i)=\frac{1}{30}$. According to Eq. {eq}`entropy`, the information content of $X$ is thus $H(X)=\avg{\log 30} =\log 30$.

We can ask "is your character Jack?", i.e. measure the observable $J = X \text{ is Jack}$, we are pretty certain that the answer will be "no". This answer will not surprise us and we'll gain very little information, only $\log \frac{30}{29}\approx 0.03$ bits, from the observation $X\ne\mbox{Jack}$. However, if the answer is "yes", which happens with probability 1/30, we'd narrow it down to a single option, and win the game. Note that the information that we'd gain is $\log 30$, which is exactly the total information content of $X$ - there's nothing more to learn. On average, we expect to gain $\frac{1}{30}\log 30+\frac{29}{30}\log\frac{30}{29}=0.21$

In contrast, if we measure the observable $W = X \text{ is a woman}$ we're certain that whatever the answer would be, we'd still have 15 options left. That is, for both possible answers, we'd gain exactly $\log 2=1$ of information, which is higher (duh) than measuring $J$.


## Bayesian interpretation
* Likelihood
* Estimating the probability of a coin toss
* Independence, conditional independence
* information, conditional information, KL divergence, mutual information

## Covariance

## Some important distributions
* Binomial
* Poisson
* Gaussian (uni+multivariate)
* Empirical
* ... add more while building the course and seeing what is needed ...


## References

```{bibliography}
:style: unsrt
```
