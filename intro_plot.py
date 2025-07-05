
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
sns.set_style(style= "whitegrid")

# Load  data into df (update the path as needed)
file_path = os.path.normpath(os.path.join(os.getcwd(), '..', 'Mall_Customers.csv'))


df = pd.read_csv(file_path)


 
def plot_age_distribution(df):
    plt.hist(df['Age'],bins=9, edgecolor='blue', color='orange')
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')  # histogram for Age distribution
    plt.show()
    


def plot_age_vs_spending_score(df):
    sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', color='orange')
    plt.title('Age vs Spending Score')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')  # scatter plot for Age vs Spending Score
    plt.show()



def plot_spending_vs_freq(df):
    sns.scatterplot(data=df, x='Spending Score (1-100)', y='Visit Frequency', color='yellow')
    plt.title(' Spending Score vs Visit Frequency')
    plt.xlabel('Spending Score (1-100)')
    plt.ylabel('Visit Frequency')
    plt.xticks(rotation=30) 
    plt.show() #scatterplot for Spending Score vs Visit Frequency to see if there is any relationship between the two
    
def plot_age_vs_freq(df):
    sns.scatterplot(data=df, x='Age',y='Visit Frequency', color= 'Pink')  
    plt.title('Age vs Frequency')
    plt.xlabel('Age')
    plt.ylabel('Visit Frequency')
    plt.xticks(rotation=60)
    plt.show()#scatterplot for Age vs Visit Frequency 




