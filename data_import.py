import pandas as pd
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
#print(df.notnull().sum())
print(df['CustomerID'].nunique())  # prints the number of unique values in the 'CustomerID' column
print(df['Gender'].nunique())
print(df['Age'].nunique())
print(df['Annual Income (k$)'].nunique())
print(df.columns)