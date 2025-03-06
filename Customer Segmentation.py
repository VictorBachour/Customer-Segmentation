import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('Online Retail.xlsx')
data.ffill(inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Quantity', 'UnitPrice']])

kmeans = KMeans(n_clusters=4,random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

plt.scatter(data['Quantity'], data['UnitPrice'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('Customer Segmentation')
plt.show()
