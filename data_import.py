import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from clustering import plot_elbow, run_kmeans, plot_clusters,avg_clusters,plot_avg,calc_mean_spending,assign_label
from Twod import plot_2d_scatter
import os



print("Current working directory:", os.getcwd())

file_path = os.path.normpath(os.path.join(os.getcwd(), '..', 'Mall_Customers.csv'))
print("File path:", file_path)

df = pd.read_csv(file_path)
#introductory visuzalizations and data preprocessing
print("Dimensions of df:", df.shape)  # prints the dimensions (rows, columns)
print("\nFirst 5 rows of df:")
print(df.head())
print(df.info())
print(df.describe())
print("Number of non-null values in each column:")
emp=df.notnull().sum()
print(f"Sum of non null valules {emp}")
print(df.isnull().sum())
print("Number of null values in each column:")
null_values = df.isnull().sum()
print(null_values)
dup=df.duplicated().sum()
print(f"Number of duplicate rows: {dup}")
sns.set_style(style= "whitegrid")
sns.countplot(data=df, x='Gender',edgecolor='black', palette='pastel')
plt.title('Customer Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count') #bar chart for Gender distribution
plt.show()

#plt.hist(df['Age'],bins=9, edgecolor= 'blue',color='orange')
#plt.title('Customer Age Distribution')
#plt.xlabel('Age')
#plt.ylabel('Count')  # histogram for Age distribution
#plt.show()

sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)',color='purple')
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')  # scatter plot for Annual Income vs Spending Score
plt.show()



sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', color='orange')
plt.title('Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')  # scatter plot for Age vs Spending Score
plt.show()

#sns.scatterplot(data=df, x='Gender', y='Spending Score (1-100)',color='yellow')
#plt.title('Gender vs Spending Score')
#plt.xlabel('Gender')
#plt.ylabel('Spending Score (1-100)')  # scatter plot for Gender vs Spending Score
#plt.show()

#sns.pairplot(df)
#plt.show()

# Data preprocessing
df_processed= df.drop(columns=['Gender']) #removed Gender column as it did not have a significant impact on spending score
scaler= StandardScaler() #create object to standardize the data
df_selected= df[['Age', 'Spending Score (1-100)']] #select columns to scale
scaled_features = scaler.fit_transform(df_selected)
#fit and transform data and adjusting to the standard normal distribution mean=0 and std=1
df_firstCluster=pd.DataFrame(scaled_features, columns=df_selected.columns)
print(df_firstCluster.head())

#Negative values for age indicates younger customer
#negative values for spending score indicates lower spending
plot_elbow(df_firstCluster, max_clusters=10)  # Plot elbow curve to find optimal number of clusters
kmeans,clusters= run_kmeans(df_firstCluster, n_clusters=3)  # Run KMeans clustering with 3 clusters and get the model

print(clusters) #prints the cluster labels for each customer such as [0, 1, 2, 0, 1, ...]
plot_clusters(df_firstCluster, clusters) # Plot the clusters
df_firstLabels=df_firstCluster.copy()

df_firstLabels['Cluster']=clusters #add cluster labels to the DataFrame
print(df_firstLabels.head())  # prints the first few rows of the DataFrame with cluster for each customer
print(df_firstLabels['Cluster'].value_counts()) # prints the count of customers in each cluster
avgClusters=avg_clusters(df_selected,clusters)
plot_avg(avgClusters)


#Creating threshold for the spending score based on the average across all clusters' spending scores
mean_value= calc_mean_spending(avgClusters)
df_selected['Category'] = df_selected.apply(lambda row: assign_label(row, mean_value, age_threshold=40), axis=1) #axis=1 applies function to rows
#Used lamda function to apply the assign_label function to each row in the dataframe so each customer is assigned a label based on their age and spending score
#aggregate customers by category
print(df_selected['Category'].value_counts(normalize=True)*100)  # prints the percentage of customers in each category
df_selected['Category'].value_counts().plot(kind='pie',y='Category',autopct='%1.1f%%',startangle=45,figsize=(8,8), colors=['lightblue', 'lightgreen', 'plum'])
plt.title('Customer Categories Distribution')
plt.legend(title='Category', loc='upper right')
plt.show()
#pie chart for Customer Categories Distribution
segment_summary=(df_selected.groupby('Category')[['Age', 'Spending Score (1-100)']].mean())
print(segment_summary)

# Group by category and calculate the mean age and spending score for each category
df_selected.groupby('Category')[['Age', 'Spending Score (1-100)']].mean().reset_index()
segment_melt=df_selected.groupby('Category')[['Age', 'Spending Score (1-100)']].mean().reset_index().melt(id_vars='Category', var_name= 'Metric', value_name='Average')
plt.figure(figsize=(12,6))
sns.barplot(data=segment_melt, x='Category', y='Average', hue='Metric', palette='pastel')
plt.title('Average Age and Spending Score by Customer Category')
plt.xlabel('Customer Category')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.show()

first_clust_csv= 'first_cluster_summary.csv'
df_selected.to_csv(first_clust_csv, index=False) #index=false to ensure that the rows are not indexed in the csv file


#2d scatterplot to show individual customers in each cluster

print("Plotting 2D scatter plot")
plot_2d_scatter(df_selected, 'Age', 'Spending Score (1-100)', kmeans, scaler=scaler, col='Category')

