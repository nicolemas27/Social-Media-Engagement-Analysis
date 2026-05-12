# 📊 Social Media Engagement Analysis & Prediction

A full end-to-end data science project analysing social media engagement patterns and predicting engagement rates using **PySpark** for large-scale EDA and a **scikit-learn** ML pipeline deployed via a **Streamlit** web app.

---

## 🗂️ Project Structure

```
├── BDA_pyspark_analysis.ipynb   # PySpark EDA + ML modelling notebook
├── app.py                       # Streamlit prediction web app
├── requirements.txt             # Python dependencies
└── README.md
```

> **Note:** The dataset (`social_media_engagement_data.csv`) is not included in this repo due to size. See [Dataset](#dataset) below.

---

## 🎯 Objectives

- Perform large-scale exploratory data analysis using **Apache Spark (PySpark)**
- Uncover engagement patterns across platforms, demographics, time, and sentiment
- Build and evaluate a **regression model** to predict post engagement rate
- Deploy an **interactive prediction tool** via Streamlit

---

## 🔍 Analysis Highlights

| Analysis Area | Key Finding |
|---|---|
| **Platform** | All four platforms show competitive average engagement rates (~0.43) |
| **Age & Platform** | Under-30s skew heavily toward Twitter; 30+ prefer LinkedIn |
| **Post Type** | Video content drives the highest average engagement |
| **Timing** | Friday evenings and Saturday nights show peak engagement |
| **Sentiment** | Positive sentiment posts outperform neutral and negative |
| **Geography** | Europe, Asia and Australia lead in regional engagement rates |

---

## 🤖 Machine Learning Pipeline

**Approach:** Two-stage modelling — PySpark MLlib for distributed training and scikit-learn for the production-ready serialised pipeline.

### Features Used
- **Numeric:** Likes, Comments, Shares, Impressions, Reach, Audience Age, Hour
- **Categorical (OHE):** Platform, Post Type, Audience Gender, Audience Interests, Sentiment, Day of Week

### PySpark Model
- `VectorAssembler` → `PolynomialExpansion (degree=2)` → `StandardScaler` → `LinearRegression`
- 3-fold `CrossValidator` with grid search over `regParam` ∈ {0.1, 0.01, 0.001} and `maxIter` ∈ {10, 50, 100}

### scikit-learn Model (production)
- `ColumnTransformer` (passthrough numerics + `OneHotEncoder` for categoricals)
- `LinearRegression` with 5-fold cross-validation
- Model serialised via `joblib` → `linear_regression_model_with_features.pkl`

---

## 🚀 Running the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (generates the `.pkl` file)
Run all cells in `BDA_pyspark_analysis.ipynb` — the final cell saves the model automatically.

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## Dataset

The dataset contains social media post-level records with the following fields:

`Post Timestamp`, `Platform`, `Post Type`, `Likes`, `Comments`, `Shares`, `Impressions`, `Reach`, `Engagement Rate`, `Audience Age`, `Audience Gender`, `Audience Location`, `Audience Interests`, `Sentiment`, `Campaign ID`, `Influencer ID`

A similar public dataset can be found on [Kaggle](https://www.kaggle.com/).

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

- **PySpark** — distributed data processing and MLlib modelling
- **pandas / matplotlib / seaborn** — EDA and visualisation
- **scikit-learn** — preprocessing pipeline and production model
- **Streamlit** — interactive web app
- **joblib** — model serialisation
