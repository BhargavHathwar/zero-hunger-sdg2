import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# -------------------------
# Data Loading and Cleaning
# -------------------------

def load_and_clean_data():
    df = pd.read_csv("GrainDemandProduction.csv")
    df.dropna(how="all", inplace=True)
    df.dropna(subset=["Element", "Region", "Year", "Millions of metric tons"], inplace=True)
    df.rename(columns={"Millions of metric tons": "Value"}, inplace=True)
    df["Element"] = df["Element"].str.strip()
    df["Region"] = df["Region"].str.strip()
    df["Sub-region"] = df["Sub-region"].str.strip()

    relevant_elements = [
        "Food grain demand",
        "Other grain demand",
        "Total grain demand",
        "Grain production",
        "Implied additional supply required"
    ]
    df = df[df["Element"].isin(relevant_elements)]
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_model_data(df):
    pivot = df.pivot_table(
        index=["Region", "Sub-region", "Year"],
        columns="Element",
        values="Value",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None
    pivot.columns = [str(c) for c in pivot.columns]

    num_cols = [c for c in pivot.columns if c not in ["Region", "Sub-region", "Year"]]
    for col in num_cols:
        pivot[col] = pivot[col].fillna(pivot[col].mean())

    # Use qcut to force balanced Low / Medium / High split into equal thirds
    implied_col = "Implied additional supply required"
    pivot["Risk_Label"] = pd.qcut(
        pivot[implied_col],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    )

    le_region = LabelEncoder()
    le_subregion = LabelEncoder()
    pivot["Region_enc"] = le_region.fit_transform(pivot["Region"])
    pivot["Sub-region_enc"] = le_subregion.fit_transform(pivot["Sub-region"])

    return pivot, le_region, le_subregion


# -------------------------
# Streamlit App
# -------------------------

st.set_page_config(page_title="Zero Hunger - Food Security Analysis", layout="wide")

st.title("Zero Hunger - SDG 2 : Food Security Risk Prediction")
st.write("This project uses the Grain Demand Production dataset to classify regions into food security risk levels using Random Forest and K-Means Clustering.")

raw_df = load_and_clean_data()
model_data, le_region, le_subregion = prepare_model_data(raw_df)


st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Random Forest", "K-Means Clustering"])


# -------------------------
# PAGE 1 : Dataset Overview
# -------------------------
if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(raw_df)

    st.subheader("Basic Statistics")
    st.dataframe(raw_df["Value"].describe().to_frame())

    st.subheader("Records per Region")
    fig, ax = plt.subplots()
    raw_df["Region"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Number of Records by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Average Grain Value by Element Type")
    fig2, ax2 = plt.subplots()
    raw_df.groupby("Element")["Value"].mean().sort_values().plot(kind="barh", ax=ax2, color="coral")
    ax2.set_title("Average Value by Element Type (Millions of metric tons)")
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Risk Label Distribution in Processed Data")
    fig3, ax3 = plt.subplots()
    model_data["Risk_Label"].value_counts().plot(kind="bar", ax=ax3, color=["green", "orange", "red"])
    ax3.set_title("Risk Label Counts")
    ax3.set_xlabel("Risk Level")
    ax3.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig3)


# -------------------------
# PAGE 2 : Random Forest
# -------------------------
elif page == "Random Forest":
    st.header("Random Forest Classifier - Food Security Risk Prediction")
    st.write("The model classifies each region-year record as Low, Medium, or High food security risk.")


    feature_cols = [
        "Food grain demand",
        "Grain production",
        "Other grain demand",
        "Total grain demand",
        "Implied additional supply required",
        "Region_enc",
        "Sub-region_enc",
        "Year"
    ]
    feature_cols = [c for c in feature_cols if c in model_data.columns]

    X = model_data[feature_cols]

    le_label = LabelEncoder()
    y = le_label.fit_transform(model_data["Risk_Label"])

    n_estimators = st.slider("Number of Trees", min_value=10, max_value=100, value=20, step=10)
    max_depth = st.slider("Max Depth", min_value=1, max_value=5, value=2)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Cross validation gives a realistic accuracy on small datasets
    cv_scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    st.metric("Cross-Validation Accuracy (3-fold)", f"{cv_scores.mean() * 100:.2f}%")
    st.write(f"Scores per fold: {[round(s*100,1) for s in cv_scores]}")

    # Train on full data for visualization
    clf.fit(X, y)
    y_pred_full = clf.predict(X)

    st.subheader("Classification Report (trained on full data)")
    report = classification_report(y, y_pred_full, target_names=le_label.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred_full)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_label.classes_,
                yticklabels=le_label.classes_, ax=ax4)
    ax4.set_title("Confusion Matrix - Random Forest")
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig4)

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig5, ax5 = plt.subplots()
    ax5.barh(importance_df["Feature"], importance_df["Importance"], color="teal")
    ax5.set_title("Feature Importances - Random Forest")
    ax5.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig5)

    st.subheader("Predict Risk for a New Entry")
    st.write("Enter values below to get a risk prediction.")

    input_vals = {}
    col1, col2 = st.columns(2)
    plain_features = [c for c in feature_cols if c not in ["Region_enc", "Sub-region_enc", "Year"]]
    for i, col in enumerate(plain_features):
        with (col1 if i % 2 == 0 else col2):
            input_vals[col] = st.number_input(col, value=float(model_data[col].mean()))

    input_vals["Year"] = st.selectbox("Year", [2022, 2032])
    input_vals["Region_enc"] = st.selectbox("Region", list(range(len(le_region.classes_))),
                                             format_func=lambda x: le_region.classes_[x])
    input_vals["Sub-region_enc"] = st.selectbox("Sub-region", list(range(len(le_subregion.classes_))),
                                                 format_func=lambda x: le_subregion.classes_[x])

    if st.button("Predict"):
        input_df = pd.DataFrame([input_vals])[feature_cols]
        pred = clf.predict(input_df)[0]
        risk = le_label.inverse_transform([pred])[0]
        color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk, "black")
        st.markdown(f"### Predicted Risk Level: :{color}[{risk}]")


# -------------------------
# PAGE 3 : K-Means Clustering
# -------------------------
elif page == "K-Means Clustering":
    st.header("K-Means Clustering - Grouping Regions by Hunger Patterns")
    st.write("K-Means groups regions with similar grain demand and production patterns together without needing labels.")

    cluster_features = [
        "Food grain demand",
        "Grain production",
        "Other grain demand",
        "Total grain demand",
        "Implied additional supply required"
    ]
    cluster_features = [c for c in cluster_features if c in model_data.columns]
    X_cluster = model_data[cluster_features]

    k = st.slider("Number of Clusters (K)", min_value=2, max_value=6, value=3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    model_data["Cluster"] = kmeans.fit_predict(X_cluster)

    st.subheader("Cluster Assignments")
    st.dataframe(model_data[["Region", "Sub-region", "Year"] + cluster_features + ["Cluster"]])

    st.subheader("Cluster Counts")
    fig6, ax6 = plt.subplots()
    model_data["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax6, color="mediumpurple")
    ax6.set_title("Number of Records per Cluster")
    ax6.set_xlabel("Cluster")
    ax6.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig6)

    st.subheader("Cluster Visualization : Food Demand vs Grain Production")
    fig7, ax7 = plt.subplots()
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    for cluster_id in sorted(model_data["Cluster"].unique()):
        subset = model_data[model_data["Cluster"] == cluster_id]
        ax7.scatter(
            subset["Food grain demand"],
            subset["Grain production"],
            label=f"Cluster {cluster_id}",
            color=colors[cluster_id % len(colors)],
            s=100
        )
    ax7.set_xlabel("Food Grain Demand (Millions of metric tons)")
    ax7.set_ylabel("Grain Production (Millions of metric tons)")
    ax7.set_title("K-Means Clusters : Demand vs Production")
    ax7.legend()
    plt.tight_layout()
    st.pyplot(fig7)

    st.subheader("Average Feature Values per Cluster")
    st.dataframe(model_data.groupby("Cluster")[cluster_features].mean().round(2))

    st.subheader("Elbow Method - Finding the Best K")
    inertias = []
    for ki in range(2, 8):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)

    fig8, ax8 = plt.subplots()
    ax8.plot(list(range(2, 8)), inertias, marker="o", color="darkorange")
    ax8.set_title("Elbow Method - Inertia vs Number of Clusters")
    ax8.set_xlabel("Number of Clusters (K)")
    ax8.set_ylabel("Inertia")
    plt.tight_layout()
    st.pyplot(fig8)
