import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def find_optimal_k(data, max_clusters=10,threshold=0.1):
    wcss=[] #alternate function to using plot_elbow if not sure how to find the optimal elbow value or number of clusters to use at which the WCSS does not drastically change
    for i in range(1, max_clusters+1):
        kmeans= KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
    rel_drop=[(wcss[i-1] - wcss[i]) / wcss[i-1] for i in range(1, len(wcss))]
    for i, drop in enumerate(rel_drop):
        if(drop<threshold):
            return i+1
    return max_clusters

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

    
def run_kmeans(data, n_clusters): #take the scaled dataframe and the number of clusters as input from the previous function in the model which also stores centroids
    """
    Run KMeans clustering on the data and return the cluster labels.
    """
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)#stores cluster labels, the number of clusters(determined by n_clusters) and centroids and how many clusters to find
    cluster_labels = kmeans.fit_predict(data) #fit the model and assign clusters to data points, note that fit_predict() will return the cluster for each customer
    return kmeans,cluster_labels

def plot_clusters(data_og, cluster_labels, xcol, ycol):
    df_plot = data_og.copy() #make a copy of the data to avoid modifying the original DataFrame
    df_plot['Cluster'] = cluster_labels# add the cluster and a new column to the DataFrame for plotting
    """
    Plot the clusters based on the cluster labels.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x=xcol, y=ycol, hue=cluster_labels, palette='viridis', s=60)
    plt.title(f'Clusters  based on {xcol} and {ycol}')
    plt.xlabel(xcol)
    plt.ylabel(ycol)
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



def plot_avg(clusters_avg,xcol,ycol): 
    #reset_index will make cluster a column instead of an index
    df_melt=clusters_avg.reset_index().melt(id_vars='Cluster', var_name= 'Feature', value_name='MeltedValue')
    plt.figure(figsize=(5,2))
    sns.barplot(data=df_melt, x='Cluster', y='MeltedValue', hue='Feature',color='Skyblue', palette='pastel')
    plt.title(f'Average {xcol} and {ycol} by Cluster')
    plt.xlabel(xcol)
    plt.xticks(rotation=75)
    plt.ylabel(ycol)
    plt.show()
    
    ## result of melt
    

    
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
        
def get_clusters_avg(df_og, cluster_col='Clusters', features=None):
    """
    Compute average values of selected features for each cluster,
    returning Spending Score first, then Visit Frequency.

    Parameters:
        df_og (DataFrame): The DataFrame containing cluster assignments.
        cluster_col (str): The column name representing cluster labels.
        features (list or None): The list of features to average. If None, auto-selects numeric columns excluding the cluster label.

    Returns:
        DataFrame: Cluster-wise average values of the selected features,
                   with Spending Score first, then Visit Frequency.
    """
    if features is None:
        features = df_og.select_dtypes(include='number').columns.difference([cluster_col])

    avg_df = df_og.groupby(cluster_col)[features].mean()
    
    # Reorder columns if both exist
    cols = avg_df.columns.tolist()
    if 'Spending Score (1-100)' in cols and 'Visit Frequency' in cols:
        reordered_cols = ['Spending Score (1-100)', 'Visit Frequency'] + [col for col in cols if col not in ['Spending Score (1-100)', 'Visit Frequency']]
        avg_df = avg_df[reordered_cols]
    
    return avg_df


def avg_vist_freq(df_og):
        return df_og['Visit Frequency'].mean()
    
def avg_spend_score(df_og):
        return df_og['Spending Score (1-100)'].mean()
    
    
    
def give_label(row, spendlow, spendhigh, visitlow, visithigh):
    """
    Assigns a behavioral label based on spending score and visit frequency quantiles.
    
    Parameters:
        row (pd.Series): A row with 'Spending Score (1-100)' and 'Visit Frequency'.
        spendlow (float): 33rd percentile of spending.
        spendhigh (float): 66th percentile of spending.
        visitlow (float): 33rd percentile of visit frequency.
        visithigh (float): 66th percentile of visit frequency.

    Returns:
        str: Label describing spending and frequency behavior.
    """
    # Visit Frequency Label
    if row['Visit Frequency'] < visitlow:
        visit_label = 'Infrequent'
    elif row['Visit Frequency'] < visithigh:
        visit_label = 'Moderate'
    else:
        visit_label = 'Frequent'

    # Spending Score Label
    if row['Spending Score (1-100)'] < spendlow:
        spend_label = 'Low'
    elif row['Spending Score (1-100)'] < spendhigh:
        spend_label = 'Moderate'
    else:
        spend_label = 'High'

    return f"{spend_label} Spender, {visit_label} Visitor"



def show_pie_chart(df,label,title,color):
   prop = df[label].value_counts(normalize=True) * 100 #create a scale for which how the values will be presented
   df[label].value_counts().plot(kind='pie',y=label,autopct='%1.1f%%',startangle=45,figsize=(8,8), colors=['lightblue', 'lightgreen', 'plum']) #creates pie chart with given parameters and values within dataframe
   plt.title(title)
   plt.legend(title= label, loc='upper right')
   plt.show()

