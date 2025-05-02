# Necessary imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import pickle

# Loading the dataset
data = pd.read_excel('Online Retail.xlsx')
data.head()

# Data Preprocessing
data.dropna(subset=["CustomerID"], inplace=True)
data = data[data["Quantity"] > 0]
data = data[data["UnitPrice"] > 0]

data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]  
latest_date = data["InvoiceDate"].max()

# Feature Engineering
customer_df = data.groupby('CustomerID').agg({
    'InvoiceDate': [
        lambda x: (latest_date - x.max()).days,   # Recency
        lambda x: x.nunique()                     # Frequency
    ],
    'InvoiceNo': 'nunique',                       # Number of invoices
    'TotalPrice': 'sum',                          # Monetary
    'StockCode': 'nunique',                       # Product variety
    'Quantity': 'sum',                            # Total items and returns
    'UnitPrice': 'mean'                           # Avg of UnitPrice
})

customer_df.columns = [
    'Recency', 'PurchaseFrequency', 'NumInvoices', 'Monetary',
    'ProductVariety', 'TotalQuantity', 'AvgUnitPrice']

customer_df.reset_index(inplace=True)

# Scaling
X = customer_df.drop(columns=['CustomerID'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit + Predicting
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Sihouette Score (KMeans) is:", silhouette_score(X_scaled, kmeans.labels_))

# Cluster Mapping
cluster_mapping = {
    0: "At-Risk Customers",
    1: "Regular Customers",
    2: "Very High Value Customer",
    3: "Anomaly",
    4: "High Value Customer"
}
customer_df['Cluster'] = customer_df['Cluster'].map(cluster_mapping)

y = customer_df['Cluster']

# Upsampling for better model's performance
customer_df['Cluster'] = customer_df['Cluster'].astype(str)
dfs = [customer_df[customer_df['Cluster'] == label] for label in customer_df['Cluster'].unique()]
max_size = max(len(df) for df in dfs)

upsampled_dfs = [
    resample(df, replace=True, n_samples=max_size, random_state=42) if len(df) < max_size else df
    for df in dfs
]

customer_df_upsampled = pd.concat(upsampled_dfs).reset_index(drop=True)

# Separate X and y
X_up = customer_df_upsampled.drop(columns=['CustomerID', 'Cluster'])
y_up = customer_df_upsampled['Cluster']

# Scale again (optional but recommended in case new samples shifted distribution)
scaler = StandardScaler()
X_up_scaled = scaler.fit_transform(X_up)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_up_scaled, y_up)

# Save model and scaler
with open('rf_model.pkl','wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler, f)

print("Done")