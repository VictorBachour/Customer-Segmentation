import numpy as np
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
data_filtered['Log_Quantity'] = np.log1p(data_filtered['Quantity'])
data_filtered['Log_UnitPrice'] = np.log1p(data_filtered['UnitPrice'])


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filtered[['Quantity', 'UnitPrice']])

kmeans = KMeans(n_clusters=4,random_state=42, n_init=10)
data_filtered.loc[:, 'Cluster'] = kmeans.fit_predict(scaled_data)

plt.scatter(data_filtered['Log_Quantity'], data_filtered['Log_UnitPrice'], c=data_filtered['Cluster'], cmap='viridis',alpha=.6)
plt.xlabel('Log Quantity')
plt.ylabel('Log UnitPrice')
plt.title('Customer Segmentation after Outlier Removal')
plt.show()
