import numpy as np
def sample_transition(mdp, x, a):
    X, A, P, c, gamma=mdp
    xlinha=np.random.choice(len(X), p=P[a][x, :])
    return (x, a, c[x,a], xlinha)

import numpy.random as rnd
def egreedy(Q, eps=0.1):
    r=rnd.random()
    if r>eps: #Choose argmin Q
        mins=np.array(np.where(Q == Q.min())).flatten()
        return rnd.choice(mins)
    else:
        return rnd.choice(range(len(Q)))

def mb_learning(M, n, qinit, Pinit, cinit):
    Q=qinit
    P=Pinit
    c=cinit
    gamma=0.99
    N=np.zeros(Q.shape)
    x=rnd.choice(range(Q.shape[0]))
    for i in range(n):
        a=egreedy(Q[x,:],eps=0.15)
        x, a, cnew, xnew = sample_transition(M, x, a)
        alpha=1/(N[x,a]+1)
        N[x,a]+=1
        c[x,a]=c[x,a]+alpha*(cnew-c[x,a])
        pi=np.zeros(len(Q))
        pi[xnew]=1
        P[a][x,:]=P[a][x,:]+alpha*(pi-P[a][x,:])
        soma=sum([P[a][x,xi]*np.min(Q[xi,:]) for xi in range(len(Q))])
        Q[x,a]=c[x,a]+gamma*soma
    return(Q,P,c)

def qlearning(M, n, qinit):
    Q=qinit
    gamma=0.99
    alpha=0.3
    x=rnd.choice(range(Q.shape[0]))
    for _ in range(n):
        a=egreedy(Q[x,:],eps=0.15)
        x, a, cnew, xnew = sample_transition(M, x, a)
        Q[x,a]=Q[x,a]+alpha*(cnew+gamma*np.min(Q[xnew])-Q[x,a])
    return Q

def sarsa(M, n, qinit):
    Q=qinit
    gamma=0.99
    alpha=0.3  
    x=rnd.choice(range(Q.shape[0]))
    a=egreedy(Q[x,:],eps=0.15)
    x, a, cnew, xnew = sample_transition(M, x, a)
    for _ in range(n):
        anew=egreedy(Q[xnew,:],eps=0.15)
        Q[x,a]=Q[x,a]+alpha*(cnew+gamma*Q[xnew, anew]-Q[x,a])    
        x, a, cnew, xnew = sample_transition(M, xnew, anew)
    return Q

# %% [markdown]
# When talking about the three methods we implemented, firstly, it is important to understand the different families of RL algorithms. In this example, we implemented a model based learning, which tries to estimate the entire model (MDP), as well as two model free learning algorithms (Q-learning and Sarsa), which attempt to only estimate the Q-function (or the cost-to-go).
# 
# When it comes to the differences between these two types of models, it is particularly interesting to look at how we update the values. In the model based learning, we take into equal consideration every iteration of our program, while in model free learning methods, we give more importance to more recent iterations (which decay exponentially backwards). This is necessary since in model free methods we only keep track of the Q function, and therefore, we do not possess a global scope of our model, hence it is essential to perform conservative estimates based on later iterations (which have accumulated more information). This allows model based learning to converge faster since it takes into consideration all iterations equally, while model free learning methods might be quicker to train.
# 
# Lastly, when it comes to the difference between Q-learning and SARSA, it lies on the fact that the Q-learning is an off-policy algorithm, while SARSA is on-policy. Essentially, this means that Q-learning is able to learn a policy different from the one he is using, while SARSA is only able to update the policy he is using. Furthermore, as seen in the plots above, SARSA can overshoot the optimal policy, if given enough time to do so. Laslty, SARSA is more averse to high costs which leads to more numerical stability.


