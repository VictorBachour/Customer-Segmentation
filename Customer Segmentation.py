import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


data = pd.read_excel('Online Retail.xlsx')
data = data.ffill()


data[['Quantity', 'UnitPrice']] = data[['Quantity', 'UnitPrice']].apply(pd.to_numeric, errors='coerce')
data.dropna(subset=['Quantity', 'UnitPrice'], inplace=True)

data_filtered = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)].copy()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filtered[['Quantity', 'UnitPrice']])

kmeans = KMeans(n_clusters=4,random_state=42, n_init=10)
data_filtered.loc[:, 'Cluster'] = kmeans.fit_predict(scaled_data)

plt.scatter(data_filtered['Quantity'], data_filtered['UnitPrice'], c=data_filtered['Cluster'], cmap='viridis',alpha=.6)
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('Customer Segmentation after Outlier Removal')
plt.show()
