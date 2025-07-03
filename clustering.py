import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def plot_elbow(data, max_clusters=10):
    """
    Plot the elbow curve to determine the optimal number of clusters.
    """
    wcss = []  # List to hold Within-Cluster Sum of Squares (WCSS) values for each k
#array will store the WCSS values for each number of clusters which consists of the sum of squared distances from each point to its assigned cluster centroid
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)  # Create a KMeans object for i clusters, use random_state for consistent results and as a starting point
        
        # KMeans algorithm works by:
    #loop will test multiple k values from 1 to max_clusters until the optimal number of clusters is found
        # Fit the KMeans model to the entire dataset
        # KMeans internally:
        #   - initializes i centroids
        #   - assigns each customer (data point) to the nearest centroid
        #   - recalculates centroids based on assigned points
        #   - repeats assignment and centroid recalculation until stable (convergence)
        #convergence is reached when the centroids are around the same and the customers stay in the same cluster
        kmeans.fit(data) #stores centroids and cluster labels for each customer in the data but doesnt return(what fit_predict does)
#each cluster is based on the average of all the assigned points to that one cluster(age, spending score average )
        # inertia_ stores the final WCSS (sum of squared distances from points to their centroids)
        wcss.append(kmeans.inertia_)  # Append WCSS for current number of cluster but also stores WCSS from all the clusters tested

    # Plot the WCSS values against number of clusters to find the "elbow"
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method for Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    
def run_kmeans(data, n_clusters): #take the scaled dataframe and the number of clusters as input from the previous function
    """
    Run KMeans clustering on the data and return the cluster labels.
    """
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)#stores cluster labels, the number of clusters(determined by n_clusters) and centroids and how many clusters to find
    cluster_labels = kmeans.fit_predict(data) #fit the model and assign clusters to data points, note that fit_predict() will return the cluster for each customer
    return kmeans,cluster_labels

def plot_clusters(data, cluster_labels):
    df_plot = data.copy() #make a copy of the data to avoid modifying the original DataFrame
    df_plot['Cluster'] = cluster_labels# add the cluster and a new column to the DataFrame for plotting
    """
    Plot the clusters based on the cluster labels.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='Age', y='Spending Score (1-100)', hue=cluster_labels, palette='viridis', s=60)
    plt.title('Clusters of Customers')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Cluster')
    plt.show()
    
    
    
def avg_clusters(original_data,cluster_labels):
    """
    Calculate the average values for each cluster.
    We use the non scaled dataframe as the scaled one will not give us much insights as they are standardized to the normal distributiin with values such as-1.23,etc
    """
    df_avg = original_data.copy()
    df_avg['Cluster'] = cluster_labels  # Add cluster labels to the DataFrame
    avg_clusters = df_avg.groupby('Cluster').mean()
    return avg_clusters
#returns data frame with the average values for each cluster but into the non scaled dataframe



def plot_avg(clusters_avg): #reset_index will make cluster a column instead of an index
    df_melt=clusters_avg.reset_index().melt(id_vars='Cluster', var_name= 'Feature', value_name='Average')
    plt.figure(figsize=(5,2))
    sns.barplot(data=df_melt, x='Cluster', y='Average', hue='Feature',color='Skyblue', palette='pastel')
    plt.title('Average Spending Score and Age by Cluster')
    plt.xlabel('Cluster')
    plt.xticks(rotation=75)
    plt.ylabel('Average Spending Score (1-100)')
    plt.show()
    
    ## result of melt
    """"
    0	Age	54.03
0	Spending Score (1-100)	36.18
1	Age	30.29
1	Spending Score (1-100)	79.82
2	Age	28.71
2	Spending Score (1-100)	35.63 so we dont have to explicitly include age because of melt, it already recognize both as distinct columns in feature
    """
   # Cluster	Feature	Average

    
def calc_mean_spending(clusters_avg):
    """_summary_

calculate the threshold for spneding score across all average clusters using the nonscaled dataframe with the average values for each cluster so the dataframe from the  avg_clusters function
        clusters_avg (_type_): _description_
    """
    mean_spending=clusters_avg['Spending Score (1-100)'].mean()
    return mean_spending
    
    

def assign_label( row, spending_threshold,age_threshold=40):
    """
    Assign labels to individual customers based on their age and spending score.
    """
    if row['Age'] < age_threshold:
        if row['Spending Score (1-100)'] < spending_threshold:
            return "Young Low Spender"
        else:
            return "Young High Spender"
    else:
        if row['Spending Score (1-100)'] < spending_threshold:
            return "Older Low Spender"
        else:
            return "Older High Spender"
        
