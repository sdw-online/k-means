"""
STAGE 1: DATA CLEANING

"""

# Import your dependencies

import pandas as pd
import numpy as np 


# Import your source file
df = pd.read_csv("D:/Projects/test_data/teams.csv", low_memory=False)

# Have a quick view of your data frame
df.head()


#Display your column keys 
df.keys()


# Let's view each distinct Season in our data 
df.Season.unique()

# Check the amount of null values in dataset 
df.isnull().sum()


# Fill null values with their mean
df = df.fillna(df.mean())


# Subset data to 2020 results
df_2020 = df[df['Season'].str.contains('2019-2020', na = False)]
prem_league_2020 = df_2020[df_2020['League'].str.contains('Premier League', na = False)]

# Drop the categorical columns (ML algorithms don't understand pure text values)
df_2020 = df_2020.drop(columns=['Player_ID', 'League_ID', 'First_Name', 'Last_Name', 'Position', 'Birth_Date','Birth_Place', 'Birth_Country', 'Nationality', 'Height', 'Weight', 'Rating','Injured', 'Team', 'League', 'Season'])



# Correlation Matrix

### Let's see the variables that have a strong/weak relationship with each other

import matplotlib.pyplot as plt
%matplotlib inline


corrmat = df_2020.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(40,40))
g = sns.heatmap(df_2020[top_corr_features].corr(), 
                annot=True, 
                cmap="RdYlGn")
				

# Engineer features using the one hot encoding technique (dummy column method)
data = pd.get_dummies(df_2020, drop_first=False)
data



"""
STAGE 2: DATA NORMALIZATION

"""

# Normalize your data so that smaller-sclaed values are not impacted by the size of larger-scaled values 

from sklearn import preprocessing

x = df_2020.iloc[:,1:].values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_norm = pd.DataFrame(x_scaled)


# Squeezing our 35+ dimensions(columns) into 2 columns using the PCA method i.e. dimensionality reduction technique 

from sklearn.decomposition import PCA

pca = PCA(n_components = 2) # 
reduced = pd.DataFrame(pca.fit_transform(x_norm))

# Create a list of names for your future data frame
names = df_2020.Full_Name.tolist()

# Rename your new data frame
reduced.columns = ['x', 'y']


"""
STAGE 3: DATA CLUSTERING 

"""


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 


# This is for Inertia (the sum sqaured of the distance between every data point in a cluster and its centroid)
WCSS = []  

# Calculate the WCSS for each k value (may be wrong explanation)
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10)
    kmeans.fit(x)
    wcss = kmeans.inertia_
    WCSS.append(wcss)
	
	
	# Elbow method
""" 
ELBOW METHOD

Purpose: to determine the optimal value for k cluster

Tune your hyperparameters appropriately e.g. n_clusters, n_init etc

"""

plt.plot(range(1,11), WCSS)
plt.title("Elbow method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS Value")
plt.show()


# Let's fit our k value
K = 3
kmeans=KMeans(n_clusters=K, init="



# c = np.random.rand(20)

plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, color="red", label="Goalkeeper")
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, color="blue", label="Defender")
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, color="green", label="Midfielder")
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, color="cyan", label="Attacker")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color="yellow",label="Centroids")
plt.show()



# reduced['cluster'] = 
# reduced['name'] = names
