import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Current working directory:", os.getcwd())

file_path = os.path.normpath(os.path.join(os.getcwd(), '..', 'Mall_Customers.csv'))
print("File path:", file_path)

df = pd.read_csv(file_path)

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
plt.hist(df['Age'],bins=9, edgecolor= 'blue',color='orange')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')  # histogram for Age distribution
plt.show()
sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)',edgecolor='yellow', palette='orange')
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')  # scatter plot for Annual Income vs Spending Score
plt.show()