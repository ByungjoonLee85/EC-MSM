from A import *

def projected_dist(x,cntr,U):
    """
    x    : given point in the cluster
    cntr : center of the cluster
    U    : basis (scaled principal components)
    """
    xmcntr = x-cntr    
    dim = len(x)
    dist = 0 
    for i in range(dim):
        x_projected = np.dot(xmcntr,U[i])/np.dot(U[i],U[i])                    
        dist += x_projected**2 
    
    # must be sqrt?
    return np.sqrt(dist)

def membership_score(x,cntr,U,sc,eps):
    """
    membership score = (1/eps)exp(-sc*projected_dist(x))
    """
    dist = projected_dist(x,cntr,U)
    ms = (1./eps)*np.exp(-sc*dist)

    # scale? why?
    return ms

def euclidean_dist(x,cntr):
    xmcntr = x-cntr
    dist = LA.norm(xmcntr)

    return dist
