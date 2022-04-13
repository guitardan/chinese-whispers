import numpy as np
import matplotlib.pyplot as plt
%matplotlib

def dist(u, v, metric):
    if metric == 'cosine':
        return 1 - (np.dot(u,v)/np.sqrt(sum(u**2)*sum(v**2))) 
    return np.sqrt(sum((u-v)**2))

def get_weight_matrix(data, dist_metric):
    W = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j:
                continue
            W[i,j] = 1/dist(data[i,:], data[j,:], dist_metric)
    return W

def telephone(data, dist_metric): # https://aclanthology.org/W06-3812.pdf
    W = get_weight_matrix(data, dist_metric)
    old_classes, new_classes = np.zeros(W.shape[0]), np.arange(W.shape[0]) # initialize each node to different classes
    while not np.array_equal(old_classes, new_classes):
        old_classes = new_classes.copy()
        for n in np.random.permutation(W.shape[0]): # shuffle nodes
            unique_classes = np.unique(new_classes)
            class_distances = np.zeros(len(unique_classes))
            # compute sum of weights to the current node for each class
            for i in range(len(unique_classes)):
                class_distances[i] = W[n, new_classes==unique_classes[i]].sum()
            new_classes[n] = unique_classes[class_distances.argmax()]
            # in case of multiple strongest classes, one is chosen randomly
            if len([class_distances == class_distances.max()]) > 1:
                new_classes[n] = np.random.choice(unique_classes[class_distances == class_distances.max()])
    return new_classes

def telephone_cluster(data, dist_metric='euclidean'):
    classes = telephone(data, dist_metric)
    while (classes == classes[0]).all(): # single cluster
        print('single cluster, retrying...')
        classes = telephone(data, dist_metric)
    return classes

x = np.array([1, 2, 3, 4, 5, 1, 2, 4, 5, 8, 9, 10, 8, 9, 11, 12, 11, 12])
y = np.array([1, 2, 3, 2, 1, 5, 4, 4, 5, 1, 2,  3, 5, 4,  4,  5,  2,  1])

c = telephone_cluster(np.vstack((x,y)).T)

plt.figure()
plt.subplot(2,1,1)
plt.scatter(x, y)
plt.subplot(2,1,2)
plt.scatter(x, y, c=c)