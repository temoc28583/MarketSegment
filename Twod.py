import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_2d_scatter(df, x_col, y_col, kmeans, scaler=None, col='Category'):
    plt.figure(figsize=(10,6))
    for value in df[col].unique(): #goes through each unique value in the specified column such as young spender,etc(does not show the categories but the graph is to show each customer's age and spending score)
        subset = df[df[col] == value] #checking like this ensures that each customer's age and spending score gets plotted as a slice of df
        plt.scatter(subset[x_col], subset[y_col], label=value, s=40, alpha=0.7) #plots customers in each category
    centroid = kmeans.cluster_centers_ #keeps  mean of z score for the centroids for each cluster in a  2d numpy array
    if scaler:
        centroids_og = scaler.inverse_transform(centroid) #inverse transforms the centroids to original non scaled values out of 100
    else:
        centroids_og = centroid #keeps centroids if the data is not scaled and returns the centroids as a 2d numpy array
    """centroids_og = 
[[28.9, 75.4],    # Cluster 0 center
 [51.6, 30.3],    # Cluster 1 center
 [27.5, 40.0]]    # Cluster 2 center
    """
        #2d numpy array of centroids based on the dimensions of the number of clusters and the number of features in the data
    x_col_og = df.columns.get_loc(x_col) #gets the index of the x_col to plot the centroids(returns 0 for Age)
    #this check is necessary to ensure that scaled and non scaled features are not plotted together 
    #gets the index of the y_col to plot the centroids
    y_col_og = df.columns.get_loc(y_col) #returns index of Spending Score
    #both these lines get the index of the x_col and y_col to plot the centroids
    #centroids_og[:, x_col_og] and centroids_og[:, y_col_og] will give us the x and y coordinates of the centroids for each cluster
    plt.scatter(centroids_og[:, x_col_og], centroids_og[:, y_col_og], c='red', marker='X', s=100, label='Centroids') #plots centroids in red with 'X'
     # = [28.9, 51.6, 27.5]
     #=[75.4, 30.3,40.0]
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"2D Scatter plot of {x_col} vs {y_col}")
    plt.grid(True)
    plt.tight_layout() #adjusts the plot to fit the figure area
    plt.show()
    