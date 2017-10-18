import numpy as np
import mxnet as mx

def batch_km(data, center, count):
    """
    Function to perform a KMeans update on a batch of data, center is the centroid
    from last iteration.

    """
    N = data.shape[0]
    K = center.shape[0]

    # update assignment
    idx = np.zeros(N, dtype=np.int)
    idx_onehot = np.zeros([N,K], dtype=np.int)
    for i in range(N):
        dist = np.inf
        ind = 0
        for j in range(K):
            temp_dist = np.linalg.norm(data[i] - center[j])
            if temp_dist < dist:
                dist = temp_dist
                ind = j
        idx[i] = ind
        idx_onehot[i][ind]=1

    # update centriod
    center_new = center
    for i in range(N):
        c = idx[i]
        count[c] += 1
        eta = 1 / count[c]
        center_new[c] = (1 - eta) * center_new[c] + eta * data[i]

    return idx_onehot, center_new, count

if __name__ == '__main__':
    nClass = 2
    data = np.array([[-1,-1],[-2,-2],[1,1],[2,2]])
    center = np.array([[-1.5,-1.5],[1.5,1.5]])
    count = 100*np.ones(nClass,dtype=np.int)
    idx_onehot, center_new, count = batch_km(data,center,count)
    a = np.zeros(center.shape)
    a=center_new
    print a
    center_new[0][0]=3
    print a