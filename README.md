# 🌾 Zero Hunger — SDG 2: Food Security Risk Analysis

An interactive web application built with **Streamlit** that analyzes global grain demand and production data to classify regions by food security risk levels — aligned with the UN's Sustainable Development Goal 2 (Zero Hunger).

---

## 📌 Overview

This project uses the **Grain Demand Production dataset** (sourced from the International Food Security Assessment) to:

- Explore grain demand, production, and supply gaps across global regions
- Classify regions as **Low / Medium / High** food security risk using a **Random Forest Classifier**
- Discover regional hunger patterns through **K-Means Clustering**
- Enable interactive prediction for custom inputs via a live Streamlit dashboard

---

## 🚀 Features

| Page | Description |
|------|-------------|
| **Dataset Overview** | Explore raw data, statistics, region distributions, and risk label breakdowns |
| **Random Forest** | Train a classifier, tune hyperparameters, view confusion matrix, feature importance, and make predictions |
| **K-Means Clustering** | Group regions by hunger patterns, visualize clusters, and use the Elbow Method to find optimal K |

---

## 🗂️ Project Structure

```
zero-hunger-sdg2/
├── app.py                          # Main Streamlit application
├── GrainDemandProduction.csv       # Dataset (grain demand, production, supply gap by region/year)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
└── Zero_Hunger_SDG2_Project.ipynb  # Jupyter notebook (exploratory analysis)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/zero-hunger-sdg2.git
   cd zero-hunger-sdg2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 📊 Dataset

**Source:** International Food Security Assessment — Grain Demand & Production Dataset

**Columns:**
- `Element` — Type of grain metric (Food grain demand, Grain production, etc.)
- `Region` — Geographic region (e.g., Asia, Sub-Saharan Africa)
- `Sub-region` — Sub-regional breakdown
- `Year` — Assessment year (2022 or 2032 projections)
- `Millions of metric tons` — Quantitative value

**Key Elements Used:**
- Food grain demand
- Other grain demand
- Total grain demand
- Grain production
- Implied additional supply required *(used to derive risk labels)*

---

## 🤖 Machine Learning

### Random Forest Classifier
- **Target:** Food security risk level — `Low`, `Medium`, or `High` (derived via quantile-based binning of implied supply gap)
- **Features:** Grain demand metrics, production, region & sub-region encodings, year
- **Evaluation:** 3-fold cross-validation accuracy
- **Interactive controls:** Number of trees, max depth, custom prediction input

### K-Means Clustering
- **Goal:** Unsupervised grouping of regions by similar grain demand/production profiles
- **Interactive controls:** Number of clusters (K = 2–6)
- **Elbow Method** chart to help select the optimal K

---

## 🌍 SDG Alignment

This project directly supports **UN SDG 2: Zero Hunger** by providing data-driven insights into:
- Which regions face the highest food security risk
- How grain production compares to demand across current (2022) and projected (2032) scenarios
- Patterns that can inform policy decisions for food distribution and agricultural investment

---

## 🙌 Acknowledgements

- Dataset: [USDA International Food Security Assessment](https://www.ers.usda.gov/publications/pub-details/?pubid=44219)
- UN SDG 2 — [Zero Hunger](https://sdgs.un.org/goals/goal2)
