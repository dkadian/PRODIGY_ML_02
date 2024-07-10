# Prodigy Infotech Internship task 2 of Machine Learning
## K-means clustering algorithm to group customers 

### Overview
The objective is to implement a K-Means clustering algorithm for grouping customers of a retail store based on their purchase history.  This clustering can provide valuable insights for personalized marketing strategies, customer targeting, and overall business decision-making. It uses the customers' annual income and spending score to create clusters and visualizes the results.

### Dataset

The analysis utilizes a dataset containing customer information, including features like 'Age,' 'Annual Income,' and 'Spending Score.' The dataset serves as the foundation for training and evaluating the clustering model.

**Dataset Link:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Technologies Used:

**1.Python:** The primary programming language for the entire solution.

**2.NumPy:** Library used for numerical operations and data handling.

**3.Matplotlib:** Library for data and image visualization, here used for visulaing results.

**4.scikit-learn:** Provides machine learning tools for building regression models, feature scaling, and polynomial feature generation.

**5.pandas:** Utilized for data manipulation and analysis.

### Code Overview:

**1.Dataset Loading:** The dataset (Mall_Customers.csv) is loaded using pandas.

**2.Feature Selection:** Two features, 'Annual Income (k$)' and 'Spending Score (1-100),' are selected for clustering.

**3.Data Scaling:** Standard scaling is applied to the selected features using StandardScaler to ensure similar scales.

**4.K-Means Clustering:** K-Means clustering is performed with the specified number of clusters (n_clusters = 5). The cluster assignments are added to the original dataset as the 'Cluster' column..

**5.Data Visualization:** The clustered data is visualized in a scatter plot using 'Annual Income (k$)' on the x-axis and 'Spending Score (1-100)' on the y-axis.Each point is color-coded based on its assigned cluster.

**6.Output Display:** The cluster assignments for each customer are displayed, showing the 'CustomerID' and the assigned 'Cluster.'

### Project Outcome

**1.Optimal Clusters:** 5 clusters were determined as optimal for customer segmentation.

**2.Insights:** The analysis revealed distinct customer segments, aiding in targeted marketing efforts and personalized customer engagement.

### Conclusion:

The code demonstrates the application of K-Means clustering to segment customers based on their annual income and spending score. The visual representation allows for the identification of distinct customer segments. Further analysis or targeted marketing strategies can be devised based on the identified clusters. The code provides a straightforward example of using clustering techniques for customer segmentation.
