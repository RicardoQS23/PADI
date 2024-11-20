# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
rnd.seed(42)

# Add your code here.
def load_chain(filename, gamma):
    pmatrix=np.load(filename)
    #Sink nodes have been solved. All that is needed is to add the teleporting.
    pmatrix=pmatrix*(1-gamma)+gamma*np.ones(pmatrix.shape)/pmatrix.shape[0]
    
    #Get the names:
    states_names=[]
    for i in range(len(pmatrix)):
        states_names.append(str(i))
    states=tuple(states_names)
    return (states, pmatrix)

def prob_trajectory(chain, trajectory):
    states, pmatrix=chain
    traj=[states.index(ti) for ti in trajectory]
    prob=1
    for i in range(len(traj)-1):
        prob*=pmatrix[traj[i],traj[i+1]]
    return prob

def stationary_dist(chain):
    states, pmatrix=chain
    eig_values,eig_vectors=np.linalg.eig(pmatrix.T)
    stationary=eig_vectors[:,0]
    return np.real(stationary/sum(stationary))

# %%
# Add your code here.
def compute_dist(chain, miu0, N):
    states, pmatrix=chain
    #Não tenho a certeza que esteja fazer a multiplicacao certa, nunca fiz multiplicacao row-matrix
    miu=np.matmul(miu0, np.linalg.matrix_power(pmatrix,N))
    return miu

# %% Activity 4 Answer

#A chain is ergodic if the limit when time tends to infinity for any given initial distribution is the stationary distribution.
#Therefore, we can test this assumption using the function above (compute_dist), and choose a very large N (to simulate the "tends to infinity"). Afterwards, we just need to simulate the chain starting from each individual state, as any other initial distribution will only be a convex combination of each initial state.
#Moreover, when looking at the experimental results above, we can see that this indeed happens, and that for all tested arbitrary initial distributions, it always converges to the stationary one.

# %%
# Add your code here.
def simulate(chain, miu0, N):
    states, pmatrix=chain
    state=rnd.choice(len(states), 1, p=miu0.flatten())[0]
    traj=[states[state]]
    for i in range(N-1):
        state=rnd.choice(len(states),1,p=pmatrix[state].flatten())[0]
        traj.append(states[state])
    return tuple(traj)


# %%
# Add your code here.

N=50000
gamma=0.01
Msmall=load_chain("example.npy",0.11)
chain=Msmall
states,pmatrix=chain
trajectory=simulate(chain,np.ones(len(states))/len(states), N)
traj=[states.index(ti) for ti in trajectory]

plt.hist(traj,bins=11,range=(0,11),density=True)
stationary=stationary_dist(chain)
plt.scatter(range(len(stationary))+np.ones(len(stationary))*0.5, stationary, color='red')
plt.legend(["Stationary Distribution","Simulated Results"])
plt.show()




