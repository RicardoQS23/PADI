import numpy as np
def load_mdp(filename, gamma):
    data=np.load(filename)
    X = data['X']
    A = data['A']
    P = data['P']
    c = data['c']
    return (X, A, P, c, gamma)

# %%
# Add your code here.
def noisy_policy(mdp, a, eps):
    X, A, P, c, g=mdp
    policy=np.zeros((len(X), len(A)))
    lista=list(range(len(A)))
    lista.pop(a)
    policy[:, lista]+=eps/(len(A)-1)
    policy[:,a]+=(1-eps)
    return policy

# %%
# Add your code here.
def evaluate_pol(mdp, policy):
    _, A ,P, c, g = mdp
    P_policy = policy[:, 0, None] * P[0]

    for a in range(1, len(A)):
        P_policy += policy[:, a, None] * P[a]
    c_policy = np.sum(policy * c, axis=1, keepdims=True)
    J_policy = np.dot(np.linalg.inv(np.eye(P_policy.shape[0])-g * P_policy), c_policy)
    return J_policy

# %%
import time
# Add your code here.
def value_iteration(mdp):
    st=time.time()
    X, A, P, c, g=mdp
    i=0
    J=np.zeros(len(X))
    Jb=np.ones(len(X))
    while np.linalg.norm(J-Jb)>10**-8:
        Q=np.zeros((len(X), len(A)))
        for a in range(len(A)):
            Q[:,a]=c[:,a]+g*np.dot(P[a], J).flatten()
        Jb=J
        J=np.min(Q, axis=1, keepdims=True)
        i+=1 
    end=time.time()
    print(f"Execution time: {round(end-st, 3)} seconds")
    print(f"N. iterations: {i}")
    return J

# %%
# Add your code here.
def policy_iteration(mdp):
    st=time.time()
    i=0
    X, A, P, c, g=mdp
    pb=np.zeros((len(X), len(A)))
    p=np.ones((len(X), len(A)))/len(A)
    while np.isclose(p, pb).all()==False:
        Q=np.zeros((len(X), len(A)))
        J=evaluate_pol(mdp, p)
        for a in range(len(A)):
            Q[:,a]=c[:,a]+g*np.dot(P[a], J).flatten()
        pb=p
        phelp=np.argmin(Q,axis=1, keepdims=True).flatten()
        p=np.zeros((len(X), len(A)))
        p[list(range(len(X))),phelp]=1
        i+=1
    end=time.time()
    print(f"Execution time: {round(end-st, 3)} seconds")
    print(f"N. iterations: {i}")
    return p

# %%
NRUNS = 100 # Do not delete this
import numpy.random as rand
# Add your code here.
def simulate(mdp, policy, x0, length):
    X, A, P, c, g=mdp
    total_cost=0
    for _ in range(NRUNS):
        cost=0
        x=x0
        for j in range(length):
            a = rand.choice(len(A), p=policy[x, :])
            cost+= (g**j) * c[x,a]
            x = rand.choice(len(X), p=P[a][x, :])
        total_cost+=cost
    total_cost/=NRUNS
    return total_cost
