# Mall Customer Segmentation using K-Means Cluster
### Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#matplotlib inline 
df = pd.read_csv('Mall_Customers.csv')
df.head()
### Explore
df.describe()
df.info()
mask = df['Spending Score (1-100)'] >50
df_score = df[mask]
df_score.head()
df_score.describe()
plt.figure(figsize=(15,6))
n=0
for x in ['Age','Annual Income (k$)', 'Spending Score (1-100)']:
    n = n+1
    plt.subplot(2,3,n)
    plt.subplots_adjust(hspace = 0.2, wspace =0.2 )
    sns.histplot(df[x],bins = 20)
    plt.title('DistPlot of {}'.format(x))
plt.show()
df_score['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Spending Score(51~100): Age Distribution');
##### Our Histogram is telling us that many of people who have spending score is greater than 50 are younger
### Count Plot of Gender
plt.figure(figsize=(15,4))
sns.countplot(y='Gender', data = df_score)
plt.title('Spending Score (51~100): Age Distribution')
plt.show()
plt.figure(figsize=(15,4))
sns.countplot(y='Gender', data = df_score)
plt.title('Spending Score (0~100): Gender Distribution')
plt.show()
### Plotting the Relation between Age, Annual Income and Spending Score
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
sns.pairplot(df[['Age','Annual Income (k$)','Spending Score (1-100)']], kind='reg')
plt.tight_layout()
plt.show()
### Distribution of values in Age, Annual Income and Spending Score according to Gender
plt.figure(1,figsize=(15,6))
for gender in ['Male','Female']:
    plt.scatter(x='Age',y='Annual Income (k$)', data= df[df['Gender']== gender], s = 200, alpha = 0.7, label = gender)
plt.xlabel('Age'),plt.ylabel('Annual Income (k$)')
plt.title('Age Vs Annual Incoem wrt Gender')
plt.legend()
plt.show()
plt.figure(1,figsize=(15,6))
for gender in['Male','Female']:
    plt.scatter(x='Annual Income (k$)', y ='Spending Score (1-100)', data = df[df['Gender']==gender], s = 200, alpha =0.7, label = gender )
plt.xlabel('Annula Income (k$)'),plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income vs Spending Score wrt Gender')
plt.legend()
plt.show()
plt.figure(1,figsize=(15,6))
n = 0
for cols in ['Age','Annual Income (k$)', 'Spending Score (1-100)']:
    n = n+1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    sns.violinplot(x=cols, y ='Gender',data= df, palette ='vlag')
    sns.swarmplot(x=cols, y = 'Gender', data= df )
    plt.ylabel('Gender' if n==1 else '')
    plt.title('Boxplots & Swarmplots' if n==2 else '')
plt.show();
### Split
x = df.iloc[:,[3,4]]
print(f"x shape {x.shape}")
x.head()
### Clustering using K-Means
#### Iterate
###### Use a for loop to build and train a K-Means model where n_clusters ranges form 2 to 13 (inclusive). Each time a model is trained. Calculate the inertia and add it to the list inertia_errors. Then calculate the silhouette score and add it to the silhouette_scores
### Segmentation  using Annual Income and Spending Scores
n_clusters = range(2,13)
inertia_errors = []
silhouette_scores = []
#Add a for loop to train model and calculate inertia, silhouette score.
for k in n_clusters:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    #train model
    model.fit(x)
    #Calculate Inertia
    inertia_errors.append(model.inertia_)
    #Calculate silhouette Score
    silhouette_scores.append(silhouette_score(x,model.labels_))
print("Inertia : ",inertia_errors[:3])
print()
print("Silhouette Score : ",silhouette_scores[:3])
### Elbow Plot
# Create a line plot of inertia_errors vs n_clusters

x_values = list(range(2,13))

plt.figure(figsize=(8,6))
sns.set(style="whitegrid") #set seaborn style

#create a line plot using matplotlib
plt.plot(x_values, inertia_errors, marker='o', linestyle='-', color = 'b')

#Add labels and title
plt.title("K-Means model: Inertia vs Number of clusters")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#Turn on grid and show plot
plt.grid(True)
plt.show
# create a line plot of silhouette scores vs n_clusters
x_values = list(range(2,13))

plt.figure(figsize=(8,6))
sns.set(style="whitegrid")

#Create a line plot using matplotlib
plt.plot(x_values, silhouette_scores, marker='o', linestyle='-', color='b')

#Add labels and Title
plt.title('K-Means Model : Silhouette Scores vs Number of clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

#turn on grid and show plot
plt.grid(True)
plt.show()
##### The best number of clusters is 5
final_model = KMeans(n_clusters=5,random_state=42,n_init=10)
final_model.fit(x)

##### In a jupyter environment , please rerun this cell to show the HTML repesentation or trust the notebook. On GitHub , the HTML representation is unable to render, please try loading this page with nbviewer.org.
labels = final_model.labels_
centroids = final_model.cluster_centers_
print(labels[:5])
print(centroids[:5])

### Communicate
#Plot "Annual Income" vs "Spending score" with final_model labels
sns.scatterplot(x=df['Annual Income (k$)'], y = df['Spending Score (1-100)'], hue=labels, palette='deep')
sns.scatterplot(
    x=centroids[:,0],
    y = centroids[:,1],
    color = 'black',
    marker ='+',
    s=500   
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score");
xgb = x.groupby(final_model.labels_).mean()
xgb
#create side by side bar chart of 'xgb'
plt.figure(figsize=(8,6))

x = [0,1,2,3,4]
x_labels = labels
income_values = xgb['Annual Income (k$)']
spending_values = xgb['Spending Score (1-100)']

bar_width = 0.35
index = range(len(x))

# Create grouped bar plot using matplotlib
plt.bar(index, income_values, bar_width, label='Annual Income')
plt.bar([i+ bar_width for i in index ], spending_values, bar_width, label='Spending score')

#Add Labels and title
plt.xlabel('Cluster')
plt.ylabel('Value')
plt.title('Annual Income and Spending Score by cluster')
plt.xticks([i+bar_width/2 for i in index],x)
plt.legend()

#show plot
plt.show()
