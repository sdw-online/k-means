import pandas as pd
import numpy as np 
from matplotlib import *
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv("D:/Projects/Python/Football/Teams/teams.csv", low_memory=False)

# Only select the columns beneficial for the algorithm
df = df[['Player_ID', 'Full_Name', 'Goals.total', 'Shots.total', 'Passes.total']]

# Group data by the players and sum up each column
df = df.groupby(['Player_ID', 'Full_Name'], as_index= False).sum()

# Sort data frame by most goals 
df = df.sort_values(by='Goals.total', ascending= False)



# Barplot: Who are the top goal scorers of all time?

plt.figure(figsize=(15,6))
sns.barplot(x=data['Full_Name'], y=data['Goals.total'], palette = "rocket")
plt.title("Goals scored")
plt.xlabel("Player Names")
plt.ylabel("Goals")
plt.show()


# K-MEans
from sklearn.cluster import KMeans
import numpy as np


X1 = df[['Goals.total', 'Passes.total']].iloc[:,:].values
elbow_range = range(1,11)
inertia = []

for i in elbow_range:
    algorithm = (KMeans(n_clusters = i,
                       init = 'k-means++',
                       n_init = 10,
                       max_iter = 300,
                       tol=0.0001,
                       random_state=100,
                       algorithm='full'))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
    
plt.plot(range(1, 11), inertia, linewidth=2, color='red', marker='8')
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia/WCSS')
plt.show()


algorithm = (KMeans(n_clusters = 4,
                   init = 'k-means++',
                   n_init = 10,
                   max_iter=300,
                   tol=0.0001,
                   random_state=100,
                   algorithm= 'full'))
algorithm.fit(X1)
c = algorithm.labels_
centroids1 = algorithm.cluster_centers_


# print(centroids1)

plt.scatter(X1[:,0], X1[:,1], c=algorithm.labels_, cmap='rainbow')
plt.scatter(algorithm.cluster_centers_[:,0], algorithm.cluster_centers_[:,1], color='black')
plt.title('Cluster of Goals')
plt.xlabel('Goals')
plt.ylabel('Passes')
plt.show()
