import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SmartCart Customer Clustering",
    page_icon="🛒",
    layout="wide"
)


# ---------------------------------------------------------
# THEME
# ---------------------------------------------------------

st.sidebar.title("SmartCart")
st.sidebar.caption("Customer segmentation dashboard")

theme = st.sidebar.radio(
    "Appearance",
    ["Light", "Dark"],
    horizontal=True,
    index=1,
    help="Switch between light and dark appearance."
)

if theme == "Dark":
    COLORS = {
        "background": "#111827",
        "surface": "#1F2937",
        "text": "#F3F4F6",
        "muted": "#9CA3AF",
        "grid": "#374151",
        "accent": "#3B82F6"
    }
else:
    COLORS = {
        "background": "#F8FAFC",
        "surface": "#FFFFFF",
        "text": "#1F2937",
        "muted": "#64748B",
        "grid": "#E2E8F0",
        "accent": "#2563EB"
    }

# Stable, distinct colours for cluster labels across all cluster charts.
CLUSTER_COLORS = [
    "#2563EB", "#EA580C", "#16A34A", "#DC2626", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#C2410C", "#475569"
]


def style_axis(ax):
    ax.set_facecolor(COLORS["surface"])
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)


st.markdown(
    f"""
    <style>
        :root {{ color-scheme: {"dark" if theme == "Dark" else "light"}; }}
        .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background-color: {COLORS["surface"]};
            border-right: 1px solid {COLORS["grid"]};
        }}
        [data-testid="stSidebar"] * {{ color: {COLORS["text"]}; }}
        .block-container {{
            max-width: 1420px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3 {{
            color: {COLORS["text"]};
            letter-spacing: -0.02em;
            font-weight: 650;
        }}
        p, label, [data-testid="stCaptionContainer"] {{
            color: {COLORS["muted"]};
        }}
        div[data-testid="stMetric"] {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["grid"]};
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: none;
        }}
        div[data-testid="stMetric"] label {{ color: {COLORS["muted"]}; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {COLORS["text"]};
            font-weight: 650;
        }}
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border: 1px solid {COLORS["grid"]};
            border-radius: 8px;
            overflow: hidden;
        }}
        div.stButton > button, div.stDownloadButton > button {{
            border-radius: 7px;
            border: 1px solid {COLORS["grid"]};
            box-shadow: none;
            font-weight: 600;
        }}
        div.stButton > button[kind="primary"] {{
            background-color: {COLORS["accent"]};
            border-color: {COLORS["accent"]};
            color: #FFFFFF;
        }}
        [data-testid="stToolbar"], #MainMenu, footer {{
            visibility: hidden;
            height: 0;
        }}
        header[data-testid="stHeader"] {{ background-color: transparent; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("SmartCart Customer Segmentation")
st.caption(
    "Explore customer groups, evaluate clustering quality, and compare purchasing behaviour."
)
st.divider()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("smartcart_customers.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Make sure smartcart_customers.csv is in the "
        "same GitHub repository folder as main.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


required_columns = [
    "Year_Birth", "Education", "Marital_Status", "Income",
    "Kidhome", "Teenhome", "Dt_Customer", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Complain", "Response"
]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR - ALGORITHM SETTINGS
# ---------------------------------------------------------

st.sidebar.info(
    f"Dataset contains {len(df):,} customer records and {len(df.columns)} columns."
)

st.sidebar.subheader("Model Settings")

algorithm = st.sidebar.selectbox(
    "Choose an algorithm",
    [
        "K-Means",
        "Agglomerative Clustering"
    ]
)

number_of_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=2,
    max_value=10,
    value=4
)

st.sidebar.caption("Choose the clustering method and number of segments.")
st.sidebar.divider()
st.sidebar.caption(
    "The same preprocessing and PCA-transformed features are used "
    "when comparing the clustering algorithms."
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

@st.cache_data(show_spinner="Preparing customer data and running clustering...")
def prepare_and_cluster(data, selected_algorithm, n_clusters):

    working_df = data.copy()

    # Fill missing income values
    working_df["Income"] = working_df["Income"].fillna(
        working_df["Income"].median()
    )

    # Feature engineering
    working_df["age"] = 2026 - working_df["Year_Birth"]

    working_df["Dt_Customer"] = pd.to_datetime(
        working_df["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    reference_date = working_df["Dt_Customer"].max()

    working_df["Customer_Tenure_Days"] = (
        reference_date - working_df["Dt_Customer"]
    ).dt.days

    working_df["Total_Spending"] = (
        working_df["MntWines"]
        + working_df["MntFruits"]
        + working_df["MntMeatProducts"]
        + working_df["MntFishProducts"]
        + working_df["MntSweetProducts"]
        + working_df["MntGoldProds"]
    )

    working_df["Total_Children"] = (
        working_df["Kidhome"] + working_df["Teenhome"]
    )

    # Group education categories
    working_df["Education"] = working_df["Education"].replace({
        "Basic": "UnderGraduate",
        "2n Cycle": "UnderGraduate",
        "Graduation": "Graduate",
        "PhD": "PostGraduate",
        "Master": "PostGraduate"
    })

    # Group relationship categories
    working_df["Relationship"] = working_df["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner",
        "Divorced": "Single",
        "Widow": "Single",
        "Alone": "Single",
        "Absurd": "Single",
        "YOLO": "Single"
    })

    # Remove original columns that are no longer needed
    columns_to_drop = [
        "ID", "Year_Birth", "Marital_Status",
        "Kidhome", "Teenhome", "Dt_Customer",
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]

    cleaned_df = working_df.drop(
        columns=[column for column in columns_to_drop if column in working_df.columns]
    )

    # Remove outliers using the same conditions as the original project
    cleaned_df = cleaned_df[
        (cleaned_df["age"] < 90)
        & (cleaned_df["Income"] < 600000)
    ].copy()

    # Fill any remaining missing numeric values
    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(
        cleaned_df[numeric_columns].median()
    )

    # Fill missing categorical values
    categorical_columns = ["Relationship", "Education"]
    for column in categorical_columns:
        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].mode().iloc[0]
        )

    # One-hot encoding
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded_values = encoder.fit_transform(
        cleaned_df[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=cleaned_df.index
    )

    model_features = pd.concat(
        [
            cleaned_df.drop(columns=categorical_columns),
            encoded_df
        ],
        axis=1
    )

    feature_columns = model_features.columns.tolist()

    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(model_features)

    # PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(scaled_features)

    # Select clustering algorithm
    if selected_algorithm == "K-Means":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = model.fit_predict(pca_features)

    elif selected_algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(
            n_clusters=n_clusters
        )
        cluster_labels = model.fit_predict(pca_features)


    result_df = cleaned_df.copy()
    result_df["Cluster"] = cluster_labels

    return (
        result_df,
        model_features,
        feature_columns,
        encoder,
        scaler,
        pca,
        model,
        pca_features,
        cluster_labels,
        reference_date
    )


try:
    (
        df_result,
        model_features,
        feature_columns,
        encoder,
        scaler,
        pca,
        clustering_model,
        X_pca,
        cluster_labels,
        reference_date
    ) = prepare_and_cluster(
        df,
        algorithm,
        number_of_clusters
    )
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


# ---------------------------------------------------------
# MODEL EVALUATION - ELBOW AND SILHOUETTE
# ---------------------------------------------------------

@st.cache_data(show_spinner="Evaluating K-Means cluster counts...")
def evaluate_kmeans(features, max_k=10):
    rows = []
    upper_k = min(max_k, len(features) - 1)

    for k in range(2, upper_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(features)

        # Silhouette is evaluated on a reproducible sample for responsiveness.
        sil = silhouette_score(
            features,
            labels,
            sample_size=min(1000, len(features)),
            random_state=42
        )

        rows.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "Silhouette Score": float(sil)
        })

    return pd.DataFrame(rows)


try:
    evaluation_df = evaluate_kmeans(X_pca, max_k=10)
except Exception as error:
    evaluation_df = pd.DataFrame()
    st.warning(f"Could not calculate the K-Means evaluation curves: {error}")


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

unique_labels = sorted(np.unique(cluster_labels).tolist())

if len(unique_labels) >= 2 and len(unique_labels) < len(cluster_labels):
    score = silhouette_score(
        X_pca,
        cluster_labels,
        sample_size=min(1000, len(X_pca)),
        random_state=42
    )
else:
    score = None

st.subheader("At a glance")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Source customers", f"{len(df):,}")

with metric2:
    st.metric("Customers analysed", f"{len(df_result):,}")

with metric3:
    st.metric("Clusters identified", len(unique_labels))

with metric4:
    st.metric(
        "Silhouette score",
        f"{score:.3f}" if score is not None else "Not available",
        help="A higher silhouette score generally indicates better-separated clusters."
    )

st.divider()

st.header("Model Evaluation")
st.caption(
    "Use the Elbow Method and Silhouette Score to assess suitable K values. "
    "These evaluation curves are calculated with K-Means on the same PCA features."
)

if not evaluation_df.empty:
    best_row = evaluation_df.loc[evaluation_df["Silhouette Score"].idxmax()]
    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.markdown("#### Elbow Method")
        st.caption("Lower inertia means tighter clusters. Look for the point where the decrease starts slowing down.")
        fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
        fig_elbow.patch.set_facecolor(COLORS["surface"])
        ax_elbow.plot(
            evaluation_df["K"],
            evaluation_df["Inertia"],
            marker="o",
            linewidth=2.5,
            color=COLORS["accent"]
        )
        ax_elbow.set_xlabel("Number of clusters (K)")
        ax_elbow.set_ylabel("Within-cluster sum of squares (Inertia)")
        ax_elbow.set_title("Elbow Curve")
        ax_elbow.set_xticks(evaluation_df["K"])
        style_axis(ax_elbow)
        st.pyplot(fig_elbow, use_container_width=True)
        plt.close(fig_elbow)

    with eval_col2:
        st.markdown("#### Silhouette Analysis")
        st.caption("Higher values indicate better-defined and better-separated clusters.")
        fig_sil, ax_sil = plt.subplots(figsize=(7, 4))
        fig_sil.patch.set_facecolor(COLORS["surface"])
        ax_sil.plot(
            evaluation_df["K"],
            evaluation_df["Silhouette Score"],
            marker="o",
            linewidth=2.5,
            color="#10B981"
        )
        ax_sil.axvline(
            best_row["K"],
            linestyle="--",
            linewidth=1.5,
            color="#F59E0B",
            label=f"Best tested K = {int(best_row['K'])}"
        )
        ax_sil.set_xlabel("Number of clusters (K)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_title("Silhouette Score by K")
        ax_sil.set_xticks(evaluation_df["K"])
        style_axis(ax_sil)
        ax_sil.legend()
        st.pyplot(fig_sil, use_container_width=True)
        plt.close(fig_sil)

    best_col1, best_col2, best_col3 = st.columns(3)
    best_col1.metric("Best tested K", int(best_row["K"]))
    best_col2.metric("Best silhouette", f"{best_row['Silhouette Score']:.3f}")
    best_col3.metric("Current model silhouette", f"{score:.3f}" if score is not None else "N/A")
    st.caption(
        "The best tested K is the K-Means result with the highest silhouette score "
        "among the evaluated values; use the Elbow Curve as a complementary visual check."
    )
else:
    st.info("Not enough data points to calculate K-Means evaluation curves.")

st.divider()


# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------

st.header("Customer Segment Profiles")

profile_columns = [
    "Income",
    "Recency",
    "Total_Spending",
    "Total_Children",
    "age",
    "Customer_Tenure_Days"
]

profile = (
    df_result
    .groupby("Cluster")[profile_columns]
    .mean()
    .round(2)
)

st.dataframe(profile, use_container_width=True)


# Assign a stable, distinct color to each cluster label across every chart.
cluster_color_map = {
    label: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
    for index, label in enumerate(unique_labels)
}


# ---------------------------------------------------------
# GRAPH 1 - CLUSTER DISTRIBUTION
# ---------------------------------------------------------

st.divider()
st.subheader("Cluster Distribution")
st.caption("Number of customers assigned to each cluster.")

cluster_counts = (
    df_result["Cluster"]
    .value_counts()
    .sort_index()
)

fig1, ax1 = plt.subplots(figsize=(8, 4))
fig1.patch.set_facecolor(COLORS["surface"])
bars = ax1.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values,
    color=[cluster_color_map[label] for label in cluster_counts.index],
    width=0.62
)

ax1.set_title("Number of Customers in Each Cluster")
ax1.set_xlabel("Cluster")
ax1.set_ylabel("Number of Customers")
style_axis(ax1)
ax1.grid(axis="y", alpha=0.35)

for bar in bars:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height()):,}",
        ha="center",
        va="bottom"
    )

st.pyplot(fig1, use_container_width=True)
plt.close(fig1)


# ---------------------------------------------------------
# GRAPH 2 - PCA CLUSTER VISUALIZATION
# ---------------------------------------------------------

st.subheader("PCA Cluster Visualization")
st.caption(
    "Each point represents a customer. PCA reduces the prepared features "
    "to two dimensions for visualization."
)

fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor(COLORS["surface"])

for label in unique_labels:
    mask = cluster_labels == label
    label_name = f"Cluster {label}"

    ax2.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=label_name,
        alpha=0.78,
        s=28,
        color=cluster_color_map[label],
        edgecolors="none"
    )

ax2.set_title(f"{algorithm} - Customer Segments")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
style_axis(ax2)
ax2.legend(frameon=True)

st.pyplot(fig2, use_container_width=True)
plt.close(fig2)


# ---------------------------------------------------------
# GRAPH 3 - INCOME VS SPENDING
# ---------------------------------------------------------

st.subheader("Income vs Total Spending")
st.caption("Compare customer income and spending behaviour across clusters.")

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor(COLORS["surface"])

for label in unique_labels:
    cluster_data = df_result[df_result["Cluster"] == label]
    label_name = f"Cluster {label}"

    ax3.scatter(
        cluster_data["Income"],
        cluster_data["Total_Spending"],
        label=label_name,
        alpha=0.78,
        s=30,
        color=cluster_color_map[label],
        edgecolors="none"
    )

ax3.set_title("Customer Income vs Total Spending")
ax3.set_xlabel("Income")
ax3.set_ylabel("Total Spending")
style_axis(ax3)
ax3.legend(frameon=True)

st.pyplot(fig3, use_container_width=True)
plt.close(fig3)


# ---------------------------------------------------------
# GRAPH 4 - SPENDING DISTRIBUTION
# ---------------------------------------------------------

st.subheader("Spending Distribution by Cluster")

spending_groups = [
    df_result.loc[df_result["Cluster"] == label, "Total_Spending"].dropna().values
    for label in unique_labels
]

spending_names = [
    f"Cluster {label}"
    for label in unique_labels
]

fig4, ax4 = plt.subplots(figsize=(9, 5))
fig4.patch.set_facecolor(COLORS["surface"])
boxplot = ax4.boxplot(
    spending_groups,
    tick_labels=spending_names,
    patch_artist=True,
    showfliers=False
)
for patch, label in zip(boxplot["boxes"], unique_labels):
    patch.set_facecolor(cluster_color_map[label])
    patch.set_alpha(0.8)
ax4.set_title("Total Spending Across Clusters")
ax4.set_xlabel("Cluster")
ax4.set_ylabel("Total Spending")
style_axis(ax4)

st.pyplot(fig4, use_container_width=True)
plt.close(fig4)


# ---------------------------------------------------------
# GRAPH 5 - CUSTOMER DISTRIBUTION PIE CHART
# ---------------------------------------------------------

st.subheader("Customer Segment Share")

fig5, ax5 = plt.subplots(figsize=(7, 5))
fig5.patch.set_facecolor(COLORS["surface"])
ax5.pie(
    cluster_counts.values,
    labels=[
        f"Cluster {label}"
        for label in cluster_counts.index
    ],
    autopct="%1.1f%%",
    startangle=90,
    colors=[cluster_color_map[label] for label in cluster_counts.index],
    wedgeprops={"edgecolor": COLORS["surface"], "linewidth": 2},
    textprops={"color": COLORS["text"]}
)
ax5.set_title("Percentage of Customers in Each Cluster")
ax5.axis("equal")

st.pyplot(fig5, use_container_width=True)
plt.close(fig5)


# ---------------------------------------------------------
# GRAPH 6 - CLUSTER PROFILE COMPARISON
# ---------------------------------------------------------

st.subheader("Cluster Behaviour Comparison")
st.caption(
    "Values are normalized column-wise to make features with different "
    "units easier to compare."
)

heatmap_columns = [
    "Income",
    "Total_Spending",
    "age",
    "Recency",
    "Total_Children",
    "Customer_Tenure_Days"
]

heatmap_data = profile[heatmap_columns].copy()

column_min = heatmap_data.min()
column_range = (heatmap_data.max() - column_min).replace(0, 1)
normalized_profile = (heatmap_data - column_min) / column_range

fig6, ax6 = plt.subplots(figsize=(11, 5))
fig6.patch.set_facecolor(COLORS["surface"])
image = ax6.imshow(
    normalized_profile.values,
    aspect="auto",
    cmap="viridis"
)

ax6.set_xticks(range(len(heatmap_columns)))
ax6.set_xticklabels(heatmap_columns, rotation=30, ha="right")
ax6.set_yticks(range(len(normalized_profile.index)))
ax6.set_yticklabels([
    f"Cluster {label}"
    for label in normalized_profile.index
])

ax6.set_title("Normalized Cluster Profile Heatmap")
ax6.tick_params(colors=COLORS["text"])
for spine in ax6.spines.values():
    spine.set_color(COLORS["grid"])
fig6.colorbar(image, ax=ax6, label="Relative value (0 to 1)")

st.pyplot(fig6, use_container_width=True)
plt.close(fig6)


# ---------------------------------------------------------
# CUSTOMER CLUSTER PREDICTION
# ---------------------------------------------------------

st.divider()
st.header("Explore a New Customer")
st.write(
    "Enter customer information to estimate which segment the customer "
    "belongs to using the selected clustering algorithm."
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

    complain = st.selectbox("Complaint", [0, 1])

    response = st.selectbox("Campaign Response", [0, 1])

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
        ["UnderGraduate", "Graduate", "PostGraduate"]
    )

with col5:
    relationship = st.selectbox(
        "Relationship",
        ["Single", "Partner"]
    )

tenure = st.number_input(
    "Customer Tenure (Days)",
    min_value=0,
    value=300
)


if st.button("🎯 Predict Customer Cluster", use_container_width=True):

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

    encoded_customer = encoder.transform(
        customer[["Relationship", "Education"]]
    )

    encoded_customer_df = pd.DataFrame(
        encoded_customer,
        columns=encoder.get_feature_names_out(
            ["Relationship", "Education"]
        )
    )

    numeric_customer = customer.drop(
        columns=["Relationship", "Education"]
    ).reset_index(drop=True)

    final_customer = pd.concat(
        [numeric_customer, encoded_customer_df],
        axis=1
    )

    final_customer = final_customer.reindex(
        columns=feature_columns,
        fill_value=0
    )

    customer_scaled = scaler.transform(final_customer)
    customer_pca = pca.transform(customer_scaled)

    if algorithm == "K-Means":
        predicted_cluster = int(clustering_model.predict(customer_pca)[0])
        prediction_method = "K-Means model prediction"
    else:
        # Agglomerative Clustering has no predict() method.
        # Assign the new customer to the nearest existing cluster centroid.
        centroids = np.array([
            X_pca[cluster_labels == label].mean(axis=0)
            for label in unique_labels
        ])

        distances = np.linalg.norm(centroids - customer_pca[0], axis=1)
        predicted_cluster = int(unique_labels[int(np.argmin(distances))])
        prediction_method = "Nearest existing cluster centroid estimate"

    st.success(
        f"Customer belongs to **Cluster {predicted_cluster}**"
    )
    st.caption(f"Assignment method: {prediction_method}")

    if predicted_cluster in profile.index:
        selected_profile = profile.loc[predicted_cluster]

        st.subheader(f"Cluster {predicted_cluster} Profile")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Average Income",
                f"₹{selected_profile['Income']:,.0f}"
            )

        with p2:
            st.metric(
                "Average Spending",
                f"{selected_profile['Total_Spending']:,.0f}"
            )

        with p3:
            st.metric(
                "Average Age",
                f"{selected_profile['age']:.0f}"
            )

        with p4:
            st.metric(
                "Average Recency",
                f"{selected_profile['Recency']:.0f}"
            )


# ---------------------------------------------------------
# DOWNLOAD AND DATA PREVIEW
# ---------------------------------------------------------

st.divider()
st.header("Customer Data")

csv_data = df_result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Clustered Data (CSV)",
    data=csv_data,
    file_name="smartcart_clustered_customers.csv",
    mime="text/csv"
)

with st.expander("View Clustered Customer Data"):
    st.dataframe(
        df_result,
        use_container_width=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()
st.caption(
    "SmartCart · Customer Intelligence · Unsupervised Machine Learning"
)
