import pandas as pd
import numpy as np 
from matplotlib import *
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Read the football teams into a data frame 
df = pd.read_csv("D:/Projects/Python/Football/Teams/teams.csv", low_memory=False)

# Only select the columns beneficial for the algorithm
df = df[['Player_ID', 'Full_Name', 'Goals.total', 'Shots.total', 'Passes.total']]

# Group data by the players and sum up each column
df = df.groupby(['Player_ID', 'Full_Name'], as_index= False).sum()

# Sort data frame by most goals 
df = df.sort_values(by='Goals.total', ascending= False)



# Barplot: Who are the top goal scorers and how many career goals have they scored?

plt.figure(figsize=(15,6))
sns.barplot(x=data['Full_Name'], y=data['Goals.total'], palette = "rocket")
plt.title("Goals scored")
plt.xlabel("Player Names")
plt.ylabel("Goals")
plt.show()




# K-Means model
from sklearn.cluster import KMeans


# Pick X 
X1 = df[['Goals.total', 'Passes.total']].iloc[:,:].values

# Find out the optimal value for k clusters with the Elbow Method 
elbow_range = range(1,11)   # Range of k between 1 and 11 
inertia = []                # This is also the WCSS (within cluster sum of squares)

# Hyperparameters in "for" loop
for i in elbow_range:
    algorithm = (KMeans(n_clusters = i,         # i is from 1 to 11
                       init = 'k-means++',
                       n_init = 10,             # the amount of times the algorithm will pick initial clusters   
                       max_iter = 300,          # maximum no of iterations is 300 per centroid initialization (selection of centroids). 
                       tol=0.0001,              # User-defined tolerance for how distant each point should be (...to be revised...)
                       random_state=100,        # 100 random numbers every time?  
                       algorithm='full'))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
    

# Create a scattrer plot for the Elbow Method 
plt.plot(range(1, 11), inertia, linewidth=2, color='red', marker='8')
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia/WCSS')
plt.show()


# The optimal value here is 4 - let's use this for the next step
algorithm = (KMeans(n_clusters = 4,
                   init = 'k-means++',
                   n_init = 10,             # the algorithm will choose initial centroids 10 times. Each pick will have its own run, with the best of these 10 picks selected as the final choice.   
                   max_iter=300,            # the algorithm will update the centroids and then reassign the data points to the correct clusters 300 times max           
                   tol=0.0001,
                   random_state=100,
                   algorithm= 'full'))
algorithm.fit(X1)
c = algorithm.labels_
centroids1 = algorithm.cluster_centers_

# Let's see how the machine has plotted the clusters 
# print(centroids1)


#  Create a final scatter plot to see the clusters within our data
plt.scatter(X1[:,0], X1[:,1], c=algorithm.labels_, cmap='rainbow')
plt.scatter(algorithm.cluster_centers_[:,0], algorithm.cluster_centers_[:,1], color='black')
plt.title('Cluster of Goals')
plt.xlabel('Goals')
plt.ylabel('Passes')
plt.show()


"""

K-means is a wonderful algorithm for animated visuals.

- Try creating a project with live animations using Python or R, also experiment wih the DBSCAN algorithm. 
- Measure the accuracy of your model to be certain your process is correct.


For more details on K-Means visit the blog below:
https://www.naftaliharris.com/blog/visualizing-k-means-clustering/ 


"""
