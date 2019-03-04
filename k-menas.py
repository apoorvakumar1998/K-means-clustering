from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('tshirt.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

# Getting the input values 
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
#scatter plotting of X and Y axis where c-"colour",s-"size"

#  Caculating Euclidean Distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 3
#T-shirt sizes=S,M and L
# X coordinates of random k centroids
C_x = np.random.randint(0, np.max(X), size=k)
# Y coordinates of random k centroids
C_y = np.random.randint(0, np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(C)

# Plotting the graph along with the random initialized Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables either 0 or 1 or 2
clusters = np.zeros(len(X))
# Error function is Distance between new centroids and old centroids
# When error function is zero,Stop finding new centroid values
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value(point) to its closest cluster
    #loop till it reaches last point
    for i in range(len(X)):
        #finding Euclidean Distance between each point with  3 centroids
        distances = dist(X[i], C)
         #argmin Returns indices of the min element of the array in a particular axis.
        cluster = np.argmin(distances)
        #storing cluster number assigned for a value X[i] in clusters variable
        clusters[i] = cluster
    # Storing the old centroid values 
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        for j in range(len(X)): 
             for j in range(len(X)):
                 if clusters[j] == i:
                      points = X[j];
        #Finding new cluster centroid by taking average of distances 
        C[i] = np.mean(points,axis=0)
    error = dist(C, C_old,None)

colors = ['r', 'g', 'b', 'y', 'c']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')



