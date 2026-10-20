# The machine learning algorithm will look at customer behaviour and Discover groups by itself. This is called unsupervised learning.

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Getz Pharma Customer Segmentation",
    page_icon="💊",
    layout="wide"
)


st.title("💊 Getz Pharma Cusotmer Segementation")

st.subheader("Unsupervised Learning with K-Means")

st.caption("")

st.info("")

# load the data

DATA_FILE = "getz_pharma_customer_segmentation_1000.csv"

df = pd.read_csv(DATA_FILE)

# QUICK DATASET OVERVIEW

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Customers",len(df))

with m2:
    st.metric("Cities", df['City'].nunique())

with m3:
    st.metric("Avg Monthly Orders", f"{df['Monthly_Orders'].mean():.1f}")

with m4:
    st.metric("Avg Monthly Spend", f"PKR {df['Monthly_Spend_PKR'].mean():,.0f}")


## create tabs

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Dataset",
        "📈 2. Choose K",
        "🤖 3. Customer Segments",
        "🧠 4. What we Learned"
    ]
)


with tab1:
    st.header("Step1 : Understand the Data")

    st.write(
        "Before using Machine Learing, we first understand what every"
        "column means. A good Data Scientist should never train a model"
        "without understanding the data"
    )

    st.dataframe(df.head(20), width ="stretch")

    st.subheader("Missing Values Check")

    missing_values = df.isna().sum()

    st.dataframe(
        missing_values.reset_index().rename(
            columns={"index": "Column", 0: "Missing_Values"}
        ),
        width = "stretch",
        hide_index=True
    )


# select the features for K-Means

features = [
    "Monthly_Orders",
    "Avg_Order_Value_PKR",
    "Products_Purchased",
    "Payment_Delay_Days",
    "Days_Since_Last_Order"
]

X = df[features]

# scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


with tab2:
    st.header("Step 2: How many groups should we make?")

    st.subheader("Elbow Method")

    st.write(
        "We can train K-Mean with different values of K ad compare"
        "Inertia: Inertia tells us how tightly customers sit around"
        "their cluster centres. Lower is better, but we do not keep adding "
        "clustering forever. We look for a bend, or 'elbow', where the"
        "improvement starts slowing down."
    )


    # we will test k from 1 to 7
    k_values = list(range(1,8))
    inertia_values = []


    # Train a small K-Means model for every K value.
    # we only save the inertia so that we can draw the elbo chart

    for k in k_values:
        elbow_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        elbow_model.fit(X_scaled)
        inertia_values.append(elbow_model.inertia_)

    elbow_df = pd.DataFrame(
        {
            "K" : k_values,
            "Inertia" : inertia_values
        }
    )

    st.line_chart(
        elbow_df,
        x="K",
        y="Inertia",
        x_label="Number of Clusters (K)",
        y_label="Inertia"
    )

    st.success(
        "For this syntetic classroom dataset, the chart has a clear bend"
        "aound K=3. That makes three clusters a sensible teaching choice."
    )

    # You guys decide this

    selected_k = st.slider(
        "Choose the number of clusters (K)",
        min_value = 2,
        max_value = 6,
        value = 3,
        step=1
    )



# Train K-Means

# n_clusters = how many groups we want

model = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

cluster_labels = model.fit_predict(X_scaled)

## add the cluster result to a copy of the data


df_clustered = df.copy()
df_clustered["Cluster"] =cluster_labels


## create a cluster summary

cluster_summary =(
    df_clustered.groupby("Cluster") 
    .agg(
        Customers=("Customer_ID", "count"),
        Monthly_Orders=("Monthly_Orders","mean"),
        Avg_Order_Value_PKR=("Avg_Order_Value_PKR","mean"),
        Product_Purchased = ("Products_Purchased", "mean"),
        Payment_Delay_Days=("Days_Since_Last_Order","mean"),
        Monthly_Spend_PKR= ("Monthly_Spend_PKR","mean"),
    )
    .round(1)
)


## Lowest spend -> Low Engagement
## Middle Spend -> Regular Customer
## Highest Spend - > High Value Loyal


if selected_k == 3:
    ordered_clusters = (
        cluster_summary["Monthly_Spend_PKR"]
        .sort_values()
        .index
        .tolist()
    )
    segment_names = {
        ordered_clusters[0] : "Low Engagement",
        ordered_clusters[1] : "Regular Customers",
        ordered_clusters[2] : "High Value loyal",

    }

    df_clustered["Segment"] = df_clustered["Cluster"].map(segment_names)

else:
    df_clustered["Segment"] = (
        "Cluster" + df_clustered["Cluster"].astype(str)
    )

    ## customer segment

with tab3:
    st.header("Step 3: Explore the Groups K-Means Discovered")
    st.subheader("How many customers are in each group?")

    segment_counts = (
        df_clustered["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        segment_counts,
        x="Segment",
        y="Customers",
        x_label="Customer Segments",
        y_label="Number of Customers"
    )

    st.subheader("Visualize the Clusters")

    st.scatter_chart(
        df_clustered,
        x="Monthly_Orders",
        y="Avg_Order_Value_PKR",
        color="Segment",
        size="Products_Purchased",
        x_label="Monthly Orders",
        y_label="Average Order Value (PKR)"
    )

    st.subheader("Cluster Summary")


    segment_summary = (
        df_clustered.groupby("Segment")
        .agg(
            Customers=("Customer_ID","count"),
            Avg_Monthly_Orders=("Monthly_Orders","mean"),
            Avg_Order_Value_PKR=("Avg_Order_Value_PKR","mean"),
            Avg_Products=("Products_Purchased","mean"),
            Avg_Payment_Delay = ("Payment_Delay_Days","mean"),
            Avg_Days_Since_Last_Order=("Days_Since_Last_Order","mean"),
            Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean"),
        )
        .round(1)
        .reset_index()
    )

    st.dataframe(segment_summary, width="stretch", hide_index=True)

    if selected_k == 3:
        st.success(
            "Great! The model has discovered three useful customer"
            "behaviour groups. Remember: we need them only AFTER studying their behaviour"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader(" 🌸 High Value Loyal")
            st.write(
                "Usually place more orders, buy more products and spend more"
            )
        with c2:
            st.subheader(" 🤨 Regular Customers")

        with c3:
            st.subheader("🤦🏻‍♂️ Low Engagement")

    st.subheader("Explore Individual Customers")

    segment_filter = st.selectbox(
        "Choose a Segment",
        options=sorted(df_clustered["Segment"].unique()),
    )

    filtered = df_clustered[
        df_clustered["Segment"] == segment_filter
    ]

    st.dataframe(filtered, width="stretch",hide_index=True)


