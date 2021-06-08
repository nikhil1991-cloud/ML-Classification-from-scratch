import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial import distance



def K_means(X,K,max_iters):
    '''This function takes in an unlabelled input array of size (N,M) with N observations and M features and labels the data into K
       clusters using K-means algorithm. The algorithm starts by randomly selecting K centroids from the observations and
       assigns clusters of data points close to the respective intial K centroids. The algorithm then updates the centroids with
       the mean of the K clusters. This process is iterated over max_iterationss unless consecutive centroids differ by zero.
       The output is a 1-D array of N labels each specifying a class for the observations.

       Input:
       
            X: Array of size (N,M). N data points and M features. Each data point belongs to R^M space.
       
            K: Number of clusters
            
            max_iters: maximum number of iterations
            
       Output:
       
            labels: Array of size (N,1) consisting of K labels for N data points.
    '''
    n_samples,n_features=X.shape
    #randomly select K centroids from n_sampels
    random_sample_idxs = np.random.choice(n_samples,K, replace=False)
    centroids = X[random_sample_idxs,:]
    #Initialize cluster list for each K
    clusters = [[] for _ in range(K)]
    for iters in range (0,max_iters):
        #Calculate distance of each point from all the centroids
        distances = distance.cdist(X,centroids, 'euclidean')
        closest_index = np.argmin(distances,axis=1)
        #Assign data points to different clusters
        for cls in range (0,len(clusters)):
            ids = np.where(closest_index==cls)[0]
            clusters[cls] = ids.tolist()
        #Assign this value to old centroid
        centroids_old=centroids
        #Update the value of centroids by taking the mean of each cluster
        centroids = np.zeros((K,n_features))
        for new_cls in range (0,len(clusters)):
            cluster_mean = np.mean(X[clusters[new_cls]],axis=0)
            centroids[new_cls] = cluster_mean
        #Convergence criteria (old_centroids=centroids)
        conv_distances = np.sum(np.sqrt(np.sum((centroids_old-centroids)**2,axis=1)))
        if conv_distances == 0.0:
            break
    #label the data as per current clusters
    labels = np.empty(n_samples)
    for clf in range (0,len(clusters)):
        labels[clusters[clf]]=clf
    return labels



#Try it on a 2-D example data set with 6 clusters
X, y = make_blobs(centers=6, n_samples=500, n_features=2, shuffle=True, random_state=40)

labels=K_means(X,6,100)

fig = plt.figure()
plt.scatter(X[:,0],X[:,1],c=labels,s=20)
cbar=plt.colorbar()
plt.xlabel('X1')
plt.ylabel('X2')
