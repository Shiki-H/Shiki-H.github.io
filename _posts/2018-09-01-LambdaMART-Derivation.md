---
layout:     post
title:      "LambdaMART Derivation"
tags:
    - machine learning
---

Last time we talked about the general workflow of learning to rank and how to implement LambdaMART in python. This time, we will go through the derivations of LambdaMART. 

## RankNet

RankNet maps an input feature vector $x\in \mathbb{R}^N$ to a number $f(x)$. For each pair of documents with different labels $d_i$ and $d_j$, we can obtain a score $s_i=f(x_i)$ and $s_j=f(x_j)$. Let $d_i\rhd d_j$ denotes the event that $d_i$ is ranked above $d_j$. We can map the scores $s_i$ and $s_j$ to a learned probability model such that $d_i$ should be ranked higher than $d_j$:
$$
\newcommand{\pder}[2][]{\frac{\partial#1}{\partial#2}}
\newcommand{\eqdef}{=\mathrel{\mathop:}}
\newcommand{\defeq}{\mathrel{\mathop:}=}
\DeclareMathOperator*{\argmin}{arg\,min}
P_{ij}=\mathbb{P}(d_i\rhd d_j)=\frac{1}{1+e^{-\sigma (s_i-s_j)}}
$$

We define the cost function using cross entropy

$$
C = -\bar{P}_{ij}logP_{ij}-(1-\bar{P}_{ij})log(1-P_{ij})
$$

where $\bar{P}_{ij}$ is the known probability that $d_i\rhd d_j$. 

Let $S_{ij}\in\{0, \pm 1\}$ be 1 if $d_i\rhd d_j$, -1 if $d_j\rhd d_i$, and 0 if $d_i$ and $d_j$ have the same rank. For simplicity, we assume that the ranking is deterministic known from labelled data. Thus, we have $\bar{P}_{ij}=\frac{1}{2}(1+S_{ij})$. Re-write the cost function, we have

$$
C = \frac{1}{2}(1-S_{ij})\sigma(s_i-s_j)+log(1+e^{-\sigma(s_i-s_j)})
$$

To update the weights of the model, we have the familiar gradient descent formula

$$
w_k \rightarrow w_k - \eta\pder[C]{w_k}
$$

where 

$$
\pder[C]{w_k} = \pder[C]{s_i}\pder[s_i]{w_k}+\pder[C]{s_j}\pder[s_j]{w_k}   \tag{1}
$$

Note that 

$$
\pder[C]{s_i}=\sigma\Big[\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}}\Big]=-\pder[C]{s_j}  \tag{2}
$$

Now, we sub Eq.(2) into Eq.(1), we can factorize the cost function to the following form:

$$
\begin{aligned}
    \pder[C]{w_k}&=\pder[C]{s_i}\pder[s_i]{w_k}+\pder[C]{s_j}\pder[s_j]{w_k} \\
        &=\sigma\Big[\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}}\Big]\Big(\pder[s_i]{w_k}-\pder[s_j]{w_k}\Big) \\
        &=\lambda_{ij}\Big(\pder[s_i]{w_k}-\pder[s_j]{w_k}\Big)
\end{aligned}
$$

where 
$$
\lambda_{ij}\defeq\pder[C]{s_i}=\sigma\Big[\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}}\Big]=-\pder[C]{s_j}    \tag{3}
$$

This approach works well if the cost function is differentiable with respect to $s_i$. However, most evaluation metrics in information retrieval such as NDCG are discontinuous as many of them need to sort the result before computing the actual value. As a result, we cannot directly use gradient descent to update the model parameters. 

## LambdaRank

The problem with RankNet is that it could not optimize information retrieval metrics directly. Here is a graphical illustration of the issue.  

![ranknet_gradient]({{ site.url }}/assets/images/posts/2018-09-01-lambdamart/ranknet_gradient.png)

In the graph above, each grey line represents an irrelevant document, while blue line represents a relevant document. In the left, the number of pairwise errors is 13. By moving the relevant document at the top down by 3 and the bottom document up by 5, we can reduce the number of errors to 11. This is represented by the black arrows. However, for most information retrieval metrics, we only care about the top $k$ entries and the gradients are represented by red arrows.  

Is there any chance we can obtain the gradients (red arrows) directly? Based on empirical results, modifying Eq.(3) by scaling the size of change of evaluation metrics (such as NDCG and MAP) which we will denote by $\lvert\Delta Z_{ij}\rvert$ given by swapping rank position of $d_i$ and $d_j$ gives good results. As a result, in LambdaRank we assume there is a **utility function** $U$ whose derivative is $\lambda_{ij}$ such that 

$$
\lambda_{ij}=\pder[U]{s_i}=-\frac{\sigma}{1+e^{-\sigma(s_i-s_j)}}|\Delta Z_{ij}|
$$

Here we need to use utility function instead of cost function because for most information retrieval metrics, higher means better, so we would like to maximize it. As a result, the weight update rule becomes

$$
w_k \rightarrow w_k + \eta\pder[U]{w_k}
$$

Let $I$ denote the set of pairs $\{i, j\}$ such that $d_i\rhd d_j$. For a given query, $\lambda_i$ associated with each $d_i$ is therefore

$$
\lambda_i=\sum_{j:\{i, j\}\in I}\lambda_{ij}-\sum_{j:\{j, i\}\in I}\lambda_{ij}
$$

Thus for each $d_i$, we can write down a utility function 

$$
U = \sum_{\{i, j\}\in I}|\Delta Z_{ij}|log(1+e^{-\sigma(s_i-s_j)})  \tag{4}
$$

### Physical Interpretation of LambdaRank

In the original paper, the authors gave a physical interpretation of LambdaRank which is noteworthy. The authors suggested that we can essentially treat each document as a point mass and $\lambda$-gradients are forces on the point mass. Positive lambda represents a push toward the top rank while negative lambda represents a push toward lower rank. We can therefore compute the net force acting on the point mass and the changes in the magnitude of the forces during training. 

## MART

Before going over LambdaMART, it is worthwhile to take a quick detour to MART. The complete name for MART is Multiple Additive Regression Tree, which as its name suggests, essentially applies gradient boosting to regression trees. In a general supervised learning problem, we are trying to find an approxiamtion $\hat{f}(x)=y$ such that 

$$
\hat{f}(x)=\argmin_{f(x)}L(y, f(x))
$$

where $L(\cdot, \cdot)$ denotes loss funtion. 

Under the settings of MART, we are looking for 

$$
\hat{f}(x)=f_M(x)=\sum^M_{m=1}T(x; \Theta_m)
$$

where $T(x; \Theta_m)$ denotes a tree model with prameters $\Theta_m$. 

With stage-wise additive modeling (for more details, refer to Elements of Statistical Learning Chapter 10), at each step in the forward step-wise procedure, we have

$$
\hat{\Theta}_m=\argmin_{\Theta_m}\sum^N_{i=1}L(y_i, f_{m-1}(x_i)+T(x_i; \Theta_m))  \tag{5}
$$

Solving the above equation exactly is quite challenging. However, we realize that this is a greedy approach. At each step, we would like $T(x_i; \Theta_m)$ to minimize Eq.(5) given $f_{m-1}$ fitted on $x_i$. Thus, $T(x_i; \Theta_m)$ is analogous to the negative gradient defined by 

$$
g_{im} = \Big[\pder[L(y_i, f(x_i)]{f(x_i)}\Big]_{f(x_i)=f_{m-1}(x_i)}
$$

As a result, we can now solve Eq.(5) with 

$$
\tilde{\Theta}_m=\argmin_{\Theta}\sum^N_{i=1}[-g_{im}-T(x_i; \Theta)]^2 \tag{6}
$$

From Eq.(6), we can see that for MART model, we do not really need to define a specific loss function. In fact, all we care about is the gradient $g_{im}$. 

## LambdaMART

Now, we can see that 

- MART is a general framework that only requires a gradient to work

- LambdaRank defines a gradient

LambdaRank and MART naturally pairs up which gives us LambdaMART. 

From Eq.(4), we can obtain

$$
\pder[U]{s_i}=\sum_{\{i, j\}\in I}\frac{-\sigma|\Delta Z_{ij}|}{1+e^{\sigma(s_i-s_j)}}=\sum_{\{i, j\}\in I}-\sigma|\Delta Z_{ij}|\rho_{ij}
$$

where $\rho_{ij}\defeq\frac{1}{1+e^{\sigma(s_i-s_j)}}$. 

Then we can also obtain

$$
\frac{\partial^2U}{\partial^2_{s_i}}=\sum_{\{i, j\}\in I}\sigma^2|\Delta Z_{ij}|\rho_{ij}(1-\rho_{ij})
$$

The Newton step-size for the $k$-th leaf of the $m$-th tree is 

$$
\gamma_{km}=\frac{\sum_{x_i\in R_{km}}\pder[U]{s_i}}{\sum_{x_i\in R_{km}}\frac{\partial^2U}{\partial^2_{s_i}}}
$$

The overall procedure of LambdaMART is therefore given by

![Algorithm]({{ site.url }}/assets/images/posts/2018-09-01-lambdamart/LambdaMART.jpg)

## References

1. [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)

2. [On the Local Optimality of LambdaRank](https://www.cs.cmu.edu/~pinard/Papers/sigirfp092-donmez.pdf)

3. [Adapting Boosting for Information Retrieval Measures](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf)

4. [Learning to Rank with Nonsmooth Cost Functions](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)

5. [A Not So Simple Introduction to LambdaMART](https://liam0205.me/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)