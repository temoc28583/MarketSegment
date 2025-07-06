 Customer Segmentation Using KMeans Clustering

This project applies KMeans clustering to identify distinct customer segments based on behavioral and demographic data. It uses features like Spending Score, Visit Frequency, and Age to uncover patterns in customer behavior, supported by data visualization and actionable marketing strategies.

---

 Objective

To use unsupervised learning to:
 Group customers into meaningful segments
 Understand behavioral patterns like spending and visit frequency- Derive strategic insights for marketing and customer engagement



Features Used
Age**
Spending Score (1–100)
Visit Frequency

Excluded Features: Gender and Income were removed due to ethical concerns and to avoid reinforcing demographic bias. See [ethics.txt]

---

 Methodology

 Data was scaled using standard normalization (mean = 0, std = 1)
 Used Elbow Method (`plot_elbow`) to find the optimal number of clusters:
   Age vs Spending Score** → k = 3
   Spending Score vs Visit Frequency** → k = 6
 Applied KMeans clustering on scaled data
 Segments labeled using thresholds and centroids
 Used:
   `plot_cluster` for scatterplots
   `show_pie_chart` for distribution visualization
   `calc_mean_spending`, `avg_visit_freq` for label assignment



Key Results

 Age vs Spending Score (k = 3 → 4 Categories)

 Segment                % of Customers 

 Younger, High Spenders 38.5%          
 Older, Low Spenders    32.0%          
 Older, High Spenders   10.0%          
 Younger, Low Spenders  10.0%          

Key Insight: Younger customers tend to spend more. Clusters were clearly separated, validating the segmentation.



Spending Score vs Visit Frequency (k = 6 → 9 Categories)

 Segment                              % of Customers 

 High Spender, Frequent Visitor       13.0%          
 Moderate Spender, Moderate Visitor   12.5%          
 High Spender, Infrequent Visitor     12.0%          
 Low Spender, Moderate Visitor        11.5%          
 Moderate Spender, Frequent Visitor   11.0%          
 Low Spender, Infrequent Visitor      10.0%          
 Low Spender, Frequent Visitor        10.0%          
 Moderate Spender, Infrequent Visitor 10.0%          
 High Spender, Moderate Visitor       10.0%          

Key Insight: Not all frequent visitors are big spenders. Frequency and spend combined revealed more nuanced behaviors than either feature alone.



 Strategic Actions Suggested

 High Spender, Frequent Visitor**: VIP loyalty programs, personalized experiences
 Moderate Spend, Moderate/Frequent Visitors**: Upsell campaigns, “level-up” rewards
 Infrequent High Spenders**: Time-sensitive re-engagement
 Low Spenders: Bundle offers, targeted nudges, limited-time promotions



 Excluded Pair: Age vs Visit Frequency

No strong relationship was found; thus, no clustering was performed on this feature pair.



 Ethics Statement

To avoid discriminatory segmentation:
 Gender and Income were excluded from the clustering process
 This aligns with ethical AI practices, minimizing bias and promoting inclusivity
 Features used (Age, Spending, Frequency) were selected to avoid targeting based on protected characteristics


Tools & Libraries

-Python (Pandas, NumPy, Scikit-learn)
 Matplotlib, Seaborn for visualization




 Files in This Repository

 `main.ipynb` – Notebook with full analysis and clustering pipeline
 `insights.txt` – Final insights and strategies
`Spending Score vs Visit Frequency Pie Chart.pdf` – Segment visualization
 `README.md` – Project overview (this file)












