# Necessary imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score

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

# Upsampling for better model performance
customer_df['Cluster'] = customer_df['Cluster'].astype(str)
dfs = [customer_df[customer_df['Cluster'] == label] for label in customer_df['Cluster'].unique()]
max_size = max(len(df) for df in dfs)

upsampled_dfs = [
    resample(df, replace=True, n_samples=max_size, random_state=42) if len(df) < max_size else df
    for df in dfs
]

customer_df_upsampled = pd.concat(upsampled_dfs).reset_index(drop=True)

# Separate X and y
X1 = customer_df_upsampled.drop(columns=['CustomerID', 'Cluster'])
y1 = customer_df_upsampled['Cluster']

# Scale again (optional but recommended in case new samples shifted distribution)
scaler = StandardScaler()
X_scaled_1 = scaler.fit_transform(X1)

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_1, y1, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Sihouette Score is:", silhouette_score(X_scaled_1, customer_df_upsampled['Cluster']))
print("Done")