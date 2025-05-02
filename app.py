import streamlit as st
import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load trained scaler and model
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('rf_model.pkl')

st.title('🛒 Customer Segmentation Prediction')
st.subheader('By Thant Thaw Tun')
st.write("Please enter customer's purchase details:")

# User input
uploaded_file = st.file_uploader("Upload transaction file (.xlsx or .csv)", type=['xlsx', 'csv'])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Feature Selection + Filtering
    data.dropna(subset=["CustomerID"], inplace=True)
    data = data[data["Quantity"] > 0]
    data = data[data["UnitPrice"] > 0]
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    latest_date = data["InvoiceDate"].max()

    # Feature Engineering
    customer_df = data.groupby('CustomerID').agg({
        'InvoiceDate': [lambda x: (latest_date - x.max()).days, lambda x: x.nunique()],
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'StockCode': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    })

    customer_df.columns = ['Recency', 'PurchaseFrequency', 'NumInvoices', 'Monetary',
                           'ProductVariety', 'TotalQuantity', 'AvgUnitPrice']
    
    customer_df.reset_index(inplace=True)

    # Spliting the dataset
    X = customer_df.drop(columns=['CustomerID'])
    X_scaled = scaler.transform(X)
    predictions = rf_model.predict(X_scaled)
    customer_df['Segment'] = predictions

    # Results
    st.header("Your Customers Data")
    st.subheader("Segmented Customers Scatter Plot:")
    
    scatter = alt.Chart(customer_df).mark_circle(size=60).encode(
        x='Recency',
        y='Monetary',
        color='Segment',
        tooltip=['CustomerID', 'Recency', 'Monetary', 'Segment']).interactive()

    st.altair_chart(scatter, use_container_width=True)
    st.subheader("Segment Distribution (Bar Chart):")
    st.bar_chart(customer_df['Segment'].value_counts())

    fig, ax = plt.subplots()
    sns.heatmap(customer_df.drop(columns=['CustomerID', 'Segment']).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Optional CSV Download
    st.subheader("Feel free to download the segmented data results!")
    csv = customer_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Segmented Results", csv, "segmented_customers.csv", "text/csv")

