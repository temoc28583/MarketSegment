import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_by_category(df, label, columns, title, xcol, ycol):
    """
    Create a bar plot showing average values of the given columns by category.
    
    Parameters:
    - df (DataFrame): The input DataFrame.
    - label (str): The column name representing category labels.
    - columns (list): The list of column names to average.
    - title (str): Title for the plot.
    - xcol (str): The column to use on the x-axis (typically the label).
    - ycol (str): The label for the y-axis.
    """
    # Step 1: Group by label and calculate mean of the selected columns
    grouped_avg = df.groupby(label)[columns].mean().reset_index()

    # Step 2: Melt for plotting
    segment = grouped_avg.melt(id_vars=label, var_name='Metric', value_name=ycol)

    # Step 3: Plot
    plt.figure(figsize=(12,6))
    sns.barplot(data=segment, x=label, y=ycol, hue='Metric', palette='pastel')
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
