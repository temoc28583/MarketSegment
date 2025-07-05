import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from intro_plot import  plot_age_distribution, plot_age_vs_spending_score, plot_age_vs_freq,plot_spending_vs_freq
from clustering import plot_elbow, run_kmeans, plot_clusters,avg_clusters,plot_avg,calc_mean_spending,assign_label,find_optimal_k,get_clusters_avg,avg_vist_freq,avg_spend_score,give_label,show_pie_chart
from Twod import plot_2d_scatter
import os
from visuals import create_by_category
np.random.seed(42)
print("Current working directory:", os.getcwd())
file_path = os.path.normpath(os.path.join(os.getcwd(), '..', 'Mall_Customers.csv'))
print("File path:", file_path)
df = pd.read_csv(file_path)

#introductory visuzalizations and data preprocessing
sns.set_style(style= "whitegrid")

# Plotting initial distributions and relationships not taking into account Gender and Income to preserve ethical integrity in segmentation
df_processed= df.drop(columns=['Gender', 'Annual Income (k$)']) #removed Gender column as it did not have a significant impact on spending score
df['Visit Frequency']=np.random.randint(0,85,size=len(df))
plot_age_distribution(df)
plot_age_vs_spending_score(df)
plot_age_vs_freq(df)
plot_spending_vs_freq(df)

# Data preprocessing

scaler= StandardScaler() #create object to standardize the data
df_selected= df[['Age', 'Spending Score (1-100)']] #select columns to scale
scaled_features = scaler.fit_transform(df_selected)
#fit and transform data and adjusting to the standard normal distribution mean=0 and std=1
df_firstCluster=pd.DataFrame(scaled_features, columns=df_selected.columns)
print(df_firstCluster.head())
#Negative values for age indicates younger customer
#negative values for spending score indicates lower spending
print("Result from plot_elbow")
plot_elbow(df_firstCluster, max_clusters=10)  # Plot elbow curve to find optimal number of clusters
kmeans,clusters= run_kmeans(df_firstCluster, n_clusters=3)  # Run KMeans clustering with 3 clusters and get the model
print(clusters) #prints the cluster labels for each customer such as [0, 1, 2, 0, 1, ...]
print("Result from plot_clusters")
plot_clusters(df_selected, clusters, 'Age', 'Spending Score (1-100)')
 # Plot the clusters
df_firstLabels=df_firstCluster.copy()
df_firstLabels['Cluster']=clusters #add cluster labels to the DataFrame
print(df_firstLabels.head())  # prints the first few rows of the DataFrame with cluster for each customer
print(df_firstLabels['Cluster'].value_counts()) # prints the count of customers in each cluster
avgClusters=avg_clusters(df_selected,clusters)
print("Result from plot_avg")
plot_avg(avgClusters, 'Age', 'Spending Score (1-100)')

#Creating threshold for the spending score based on the average across all clusters' spending scores
mean_value= calc_mean_spending(avgClusters)
df_selected['Category'] = df_selected.apply(lambda row: assign_label(row, mean_value, age_threshold=40), axis=1) #axis=1 applies function to rows
#Used lamda function to apply the assign_label function to each row in the dataframe so each customer is assigned a label based on their age and spending score
#aggregate customers by category
print("Pie chart for Age vs Spending Score Distribution")
colors=['lightblue','lightgreen', 'plum']
show_pie_chart(df_selected, label='Category', title='Customer Categories Distribution',color=colors)


print("Cluster Means for Age vs Spending Score")
segment_summary=(df_selected.groupby('Category')[['Age', 'Spending Score (1-100)']].mean())
print(segment_summary)

# Group by category and calculate the mean age and spending score for each category

print("Average age and spending score for each category Age vs Spending Score")
create_by_category(
    df_selected,
    label='Category',
    columns=['Age', 'Spending Score (1-100)'],
    title='Average Age and Spending Score by Category',
    xcol='Customer Category',
    ycol='Average Value'
)

first_clust_csv= 'first_cluster_summary.csv'
df_selected.to_csv(first_clust_csv, index=False) #index=false to ensure that the rows are not indexed in the csv file

#2d scatterplot to show individual customers in each cluster
print("2D scatter plot for Age vs Spending Score")

print("Plotting 2D scatter plot")
plot_2d_scatter(df_selected, 'Age', 'Spending Score (1-100)', kmeans, scaler=scaler, col='Category')

#Creating new cluster but with visit frequency and spending score
df_frequency= df[['Visit Frequency', 'Spending Score (1-100)']]
scaled_score_vs_freq=scaler.fit_transform(df_frequency) #fitting the dataframe to standard normal distribution
df_second_cluster=pd.DataFrame(scaled_score_vs_freq, columns=df_frequency.columns)
#used the dataframe with the selected columns to keep the columns in the scaled dataframe except with values adjusted to the normal distribution
optimal_k=find_optimal_k(df_second_cluster,max_clusters=10, threshold=0.1) #finds the number of clusters to use similar to plot_elbow
print("Plot_elbow for Spending Score and Visit Frequency")
plot_elbow(df_second_cluster, max_clusters=10) #returns a graph to show the ideal number of clusters used
kmeans_behavior, cluster_behavior= run_kmeans(df_second_cluster,optimal_k)#stores clusters associated with each customer
df_frequency['Clusters'] = cluster_behavior#creates a new column with the clusters
print("Plot clusters for Spending Score and Visit Frequency")
plot_clusters(df_frequency, cluster_behavior, 'Spending Score (1-100)', 'Visit Frequency') #plots clusters for each customer with their spending score and visit frequency
avg_age_vs_freq=avg_clusters(df_frequency,cluster_behavior)
print('Cluster for each average spending score and Visit Frequency')
plot_avg(avg_age_vs_freq, 'Spending Score (1-100)', 'Visit Frequency') #plots average values of the spending score and visit frequency by cluster

'Avg Spending Score and Visit Frequency by Clusters'

avg_spend_freq = get_clusters_avg(df_frequency, features=['Visit Frequency', 'Spending Score (1-100)'])
print(avg_spend_freq)
visit_threshold=avg_vist_freq(df_frequency) #returns average visit frequencies across all customers using the raw dataframe
spend_threshold=avg_spend_score(df_frequency)#returns average spending score across all customers using the raw dataframe
# Step 1: Calculate quantile thresholds to consider moderate spenders
spend_low, spend_high = df_frequency['Spending Score (1-100)'].quantile([0.33, 0.66])
visit_low, visit_high = df_frequency['Visit Frequency'].quantile([0.33, 0.66])

# Step 2: Assign behavioral segment to each customer based on quantile thresholds using apply(using the lambda function within apply will assign the label to all the rows,per customer)
df_frequency['Customer Category'] = df_frequency.apply(
    lambda row: give_label(row, spend_low, spend_high, visit_low, visit_high),
    axis=1
)

# Create pie chart based on label distribution
colors = ['lightblue', 'lightgreen', 'plum', 'orange', 'salmon', 'violet', 'lightcoral', 'lightcyan', 'khaki']
show_pie_chart(df_frequency, label='Customer Category', title='Spending Score vs Visit Frequency Distribution', color=colors)
#contains 9 different distribitions despite 6 clusters this is because we noticed that some of the clusters had overlapped and thus seeked to accomdate to "moderate spenders" as well to avoid underfitting the customers and clusters
print("Bar chart to show the average visting frequency and average spending score by category")
create_by_category(df_frequency,'Customer Category', ['Spending Score (1-100)','Visit Frequency'],'Customer Category by Average Spending Score and Visit Frequency', 'Spending Score (1-100)', 'Average Value')
print("2D Chart showing the clusters against the Visit Frequency and spending score") 
plot_2d_scatter(df_frequency, 'Spending Score (1-100)', 'Visit Frequency', kmeans=kmeans_behavior, scaler=scaler, col='Customer Category')
#creating 2D scatter chart to show the centroids for each cluster along with all the customers

