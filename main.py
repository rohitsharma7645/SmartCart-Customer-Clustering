import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="SmartCart Customer Clustering",
    page_icon="🛒",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🛒 SmartCart Customer Clustering System")

st.write(
    "This application segments customers into different groups "
    "using K-Means clustering based on customer demographics, "
    "purchasing behavior and engagement."
)

st.divider()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("smartcart_customers.csv")


df = load_data()


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

@st.cache_resource
def train_model(data):

    df = data.copy()

    # -----------------------------
    # Missing value handling
    # -----------------------------

    df["Income"] = df["Income"].fillna(df["Income"].mean())

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df["age"] = 2026 - df["Year_Birth"]

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True
    )

    ref_date = df["Dt_Customer"].max()

    df["Customer_Tenure_Days"] = (
        ref_date - df["Dt_Customer"]
    ).dt.days

    df["Total_Spending"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    df["Total_Children"] = (
        df["Kidhome"] + df["Teenhome"]
    )

    # -----------------------------
    # Education grouping
    # -----------------------------

    df["Education"] = df["Education"].replace({
        "Basic": "UnderGraduate",
        "2n Cycle": "UnderGraduate",
        "Graduation": "Graduate",
        "PhD": "PostGraduate",
        "Master": "PostGraduate"
    })

    # -----------------------------
    # Relationship grouping
    # -----------------------------

    df["Relationship"] = df["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner",
        "Divorced": "Single",
        "Widow": "Single",
        "Alone": "Single",
        "Absurd": "Single",
        "YOLO": "Single"
    })

    # -----------------------------
    # Drop unnecessary columns
    # -----------------------------

    df_cleaned = df.drop(
        columns=[
            "ID",
            "Year_Birth",
            "Marital_Status",
            "Kidhome",
            "Teenhome",
            "Dt_Customer",
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds"
        ]
    )

    # -----------------------------
    # Outlier removal
    # -----------------------------

    df_cleaned = df_cleaned[
        (df_cleaned["age"] < 90)
        & (df_cleaned["Income"] < 600000)
    ].copy()

    # -----------------------------
    # One Hot Encoding
    # -----------------------------

    cat_cols = [
        "Relationship",
        "Education"
    ]

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded = ohe.fit_transform(
        df_cleaned[cat_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df_cleaned.index
    )

    df_encoded = pd.concat(
        [
            df_cleaned.drop(columns=cat_cols),
            encoded_df
        ],
        axis=1
    )

    # -----------------------------
    # Scaling
    # -----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        df_encoded
    )

    # -----------------------------
    # PCA
    # -----------------------------

    pca = PCA(
        n_components=3
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    # -----------------------------
    # K-Means
    # -----------------------------

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_pca
    )

    df_result = df_cleaned.copy()

    df_result["Cluster"] = labels

    return (
        df_result,
        ohe,
        scaler,
        pca,
        kmeans,
        df_encoded.columns.tolist(),
        ref_date
    )


(
    df_result,
    ohe,
    scaler,
    pca,
    kmeans,
    feature_columns,
    ref_date
) = train_model(df)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("SmartCart")

st.sidebar.write(
    "Customer segmentation using K-Means clustering."
)

st.sidebar.info(
    "The model divides customers into 4 clusters."
)


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.subheader("📊 Customer Segmentation Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Original Customers",
        len(df)
    )

with col2:
    st.metric(
        "Customers Used",
        len(df_result)
    )

with col3:
    st.metric(
        "Clusters",
        4
    )

with col4:
    st.metric(
        "Features",
        len(feature_columns)
    )


st.divider()


# ---------------------------------------------------------
# CLUSTER DISTRIBUTION
# ---------------------------------------------------------

st.subheader("📈 Cluster Distribution")

cluster_counts = (
    df_result["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_df = pd.DataFrame({
    "Cluster": cluster_counts.index.astype(str),
    "Customers": cluster_counts.values
})

st.bar_chart(
    cluster_df.set_index("Cluster")
)


# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------

st.subheader("👥 Cluster Profile")

profile = (
    df_result
    .groupby("Cluster")
    .agg({
        "Income": "mean",
        "Recency": "mean",
        "Total_Spending": "mean",
        "Total_Children": "mean",
        "age": "mean",
        "Customer_Tenure_Days": "mean"
    })
    .round(2)
)

st.dataframe(
    profile,
    use_container_width=True
)


st.divider()


# ---------------------------------------------------------
# CUSTOMER PREDICTION
# ---------------------------------------------------------

st.subheader("🔍 Find Customer Segment")

st.write(
    "Enter customer information below to determine "
    "which customer cluster they belong to."
)


col1, col2, col3 = st.columns(3)

with col1:

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=89,
        value=40
    )

    recency = st.number_input(
        "Recency",
        min_value=0,
        value=30
    )

    total_children = st.number_input(
        "Total Children",
        min_value=0,
        value=1
    )


with col2:

    deals = st.number_input(
        "Deals Purchases",
        min_value=0,
        value=2
    )

    web_purchases = st.number_input(
        "Web Purchases",
        min_value=0,
        value=4
    )

    catalog_purchases = st.number_input(
        "Catalog Purchases",
        min_value=0,
        value=3
    )

    store_purchases = st.number_input(
        "Store Purchases",
        min_value=0,
        value=5
    )


with col3:

    web_visits = st.number_input(
        "Web Visits / Month",
        min_value=0,
        value=5
    )

    complain = st.selectbox(
        "Complaint",
        [0, 1]
    )

    response = st.selectbox(
        "Campaign Response",
        [0, 1]
    )

    total_spending = st.number_input(
        "Total Spending",
        min_value=0.0,
        value=500.0,
        step=50.0
    )


col4, col5 = st.columns(2)

with col4:

    education = st.selectbox(
        "Education",
        [
            "UnderGraduate",
            "Graduate",
            "PostGraduate"
        ]
    )


with col5:

    relationship = st.selectbox(
        "Relationship",
        [
            "Single",
            "Partner"
        ]
    )


# ---------------------------------------------------------
# TENURE
# ---------------------------------------------------------

tenure = st.number_input(
    "Customer Tenure (Days)",
    min_value=0,
    value=300
)


# ---------------------------------------------------------
# PREDICT CLUSTER
# ---------------------------------------------------------

if st.button(
    "🎯 Predict Customer Cluster",
    use_container_width=True
):

    # -----------------------------
    # Create customer dataframe
    # -----------------------------

    customer = pd.DataFrame({
        "Income": [income],
        "Recency": [recency],
        "NumDealsPurchases": [deals],
        "NumWebPurchases": [web_purchases],
        "NumCatalogPurchases": [catalog_purchases],
        "NumStorePurchases": [store_purchases],
        "NumWebVisitsMonth": [web_visits],
        "Complain": [complain],
        "Response": [response],
        "age": [age],
        "Customer_Tenure_Days": [tenure],
        "Total_Spending": [total_spending],
        "Total_Children": [total_children],
        "Relationship": [relationship],
        "Education": [education]
    })

    # -----------------------------
    # Encode categorical values
    # -----------------------------

    encoded_customer = ohe.transform(
        customer[
            ["Relationship", "Education"]
        ]
    )

    encoded_customer_df = pd.DataFrame(
        encoded_customer,
        columns=ohe.get_feature_names_out(
            ["Relationship", "Education"]
        )
    )

    numeric_customer = customer.drop(
        columns=[
            "Relationship",
            "Education"
        ]
    ).reset_index(drop=True)

    final_customer = pd.concat(
        [
            numeric_customer,
            encoded_customer_df
        ],
        axis=1
    )

    # Make sure feature order matches training data

    final_customer = final_customer.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # -----------------------------
    # Scale
    # -----------------------------

    customer_scaled = scaler.transform(
        final_customer
    )

    # -----------------------------
    # PCA
    # -----------------------------

    customer_pca = pca.transform(
        customer_scaled
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    cluster = kmeans.predict(
        customer_pca
    )[0]

    st.success(
        f"Customer belongs to **Cluster {cluster}**"
    )

    # -----------------------------
    # Display cluster information
    # -----------------------------

    cluster_profile = profile.loc[
        cluster
    ]

    st.subheader(
        f"📌 Cluster {cluster} Profile"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Average Income",
            f"₹{cluster_profile['Income']:,.0f}"
        )

    with c2:
        st.metric(
            "Average Spending",
            f"{cluster_profile['Total_Spending']:,.0f}"
        )

    with c3:
        st.metric(
            "Average Age",
            f"{cluster_profile['age']:.0f}"
        )

    with c4:
        st.metric(
            "Average Recency",
            f"{cluster_profile['Recency']:.0f}"
        )


st.divider()


# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------

with st.expander("📋 View Clustered Customer Data"):

    st.dataframe(
        df_result,
        use_container_width=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.caption(
    "SmartCart Customer Clustering | "
    "Machine Learning Customer Segmentation"
)