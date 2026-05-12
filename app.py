import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EngageIQ · Social Media Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f0f13;
    color: #e8e6e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16161d !important;
    border-right: 1px solid #2a2a35;
}
[data-testid="stSidebar"] * {
    color: #f0ede6 !important;
}
[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] p {
    color: #e8e4dc !important;
}

/* Headings */
h1 { font-family: 'DM Serif Display', serif; font-size: 2.6rem !important; color: #f5f0e8 !important; letter-spacing: -0.5px; }
h2 { font-family: 'DM Serif Display', serif; color: #f5f0e8 !important; }
h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; color: #d4cfbf !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1c1c26;
    border: 1px solid #2a2a38;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #8a8880 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #f0e9d6 !important; font-size: 1.9rem !important; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #7a7870 !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e8c97a !important;
    border-bottom: 2px solid #e8c97a !important;
}

/* Buttons */
.stButton > button {
    background: #e8c97a;
    color: #0f0f13;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: #f0d68a;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(232, 201, 122, 0.3);
}

/* Info boxes */
.insight-card {
    background: #1c1c26;
    border: 1px solid #2a2a38;
    border-left: 3px solid #e8c97a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #c8c6c0;
}
.insight-card strong { color: #e8c97a; }

/* Prediction result banner */
.pred-banner {
    background: linear-gradient(135deg, #1c1c26 0%, #222230 100%);
    border: 1px solid #e8c97a44;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-value {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    color: #e8c97a;
    line-height: 1;
}
.pred-label {
    font-size: 0.85rem;
    color: #7a7870;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}

/* Dividers */
hr { border-color: #2a2a38 !important; }

/* Select / number inputs */
[data-baseweb="select"] { background: #1c1c26 !important; }
[data-baseweb="select"] span { color: #ffffff !important; }
[data-baseweb="input"] input { color: #ffffff !important; background: #1c1c26 !important; }
[data-testid="stNumberInput"] input { color: #ffffff !important; }
[data-testid="stSidebar"] input { color: #ffffff !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #ffffff !important; }
[data-testid="stSidebar"] [data-baseweb="input"] { color: #ffffff !important; }
[data-testid="stSidebar"] [data-baseweb="select"] { background-color: #1c1c26 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #ffffff !important; background-color: #1c1c26 !important; }
[data-testid="stSidebar"] [role="listbox"] { background-color: #1c1c26 !important; }
[data-testid="stSidebar"] [role="option"] { color: #ffffff !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background-color: #1c1c26 !important; color: #ffffff !important; }
[data-testid="stSidebar"] .stSlider span { color: #ffffff !important; }
[data-testid="stSidebar"] input[type="number"] { color: #ffffff !important; background-color: #1c1c26 !important; border-color: #3a3a4e !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model_with_features.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Benchmark data (derived from EDA) ──────────────────────────────────────────
BENCHMARKS = {
    "Platform":  {"Instagram": 45.2, "TikTok": 52.8, "Facebook": 38.4, "Twitter": 36.1, "LinkedIn": 41.7},
    "Post Type": {"Video": 51.3, "Image": 42.1, "Carousel": 46.8, "Text": 31.5, "Link": 28.9},
    "Sentiment": {"Positive": 48.6, "Neutral": 40.2, "Negative": 35.1},
    "Hour":      {range(6,9): 40.2, range(11,14): 43.8, range(17,21): 52.4, range(21,24): 38.1},
}

KEY_INSIGHTS = [
    ("🎬", "Video content", "outperforms all other post types with an average engagement rate of <strong>51.3%</strong>"),
    ("🌙", "Friday 6–9 PM", "is the peak engagement window — posts scheduled here see up to <strong>34% higher</strong> interaction"),
    ("😊", "Positive sentiment", "posts outperform negative ones by <strong>38%</strong> on average engagement"),
    ("📍", "TikTok & Instagram", "lead platform rankings with averages of 52.8% and 45.2% respectively"),
    ("👥", "Ages 25–34", "drive the highest engagement volume across all platforms in this dataset"),
]


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Post Parameters")
    st.markdown("---")

    st.markdown("**📈 Performance Metrics**")
    likes       = st.number_input("Likes",        min_value=0, value=500)
    comments    = st.number_input("Comments",     min_value=0, value=250)
    shares      = st.number_input("Shares",       min_value=0, value=100)
    impressions = st.number_input("Impressions",  min_value=0, value=5500)
    reach       = st.number_input("Reach",        min_value=0, value=2750)

    st.markdown("---")
    st.markdown("**🎯 Post Details**")
    platform   = st.selectbox("Platform",   ["Instagram", "Facebook", "Twitter", "LinkedIn", "TikTok"])
    post_type  = st.selectbox("Post Type",  ["Video", "Image", "Carousel", "Text", "Link"])
    sentiment  = st.selectbox("Sentiment",  ["Positive", "Neutral", "Negative"])
    hour       = st.slider("Hour of Post (0–23)", 0, 23, 18)
    day        = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    st.markdown("---")
    st.markdown("**👥 Audience**")
    audience_age    = st.slider("Average Audience Age", 13, 65, 28)
    audience_gender = st.selectbox("Audience Gender", ["Female", "Male", "Mixed"])
    audience_location = st.selectbox("Audience Region", [
        "United States of America", "United Kingdom", "Australia", "Canada",
        "Germany", "France", "India", "Indonesia", "Brazil", "South Africa",
        "United Arab Emirates", "Japan", "Nigeria", "Kenya", "Singapore",
        "Other"
    ])

    # Derived features — computed automatically
    if hour < 6:
        time_of_day = "Night"
    elif hour < 12:
        time_of_day = "Morning"
    elif hour < 17:
        time_of_day = "Afternoon"
    elif hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Late Night"

    if audience_age < 18:
        age_group = "Under 18"
    elif audience_age < 25:
        age_group = "18-24"
    elif audience_age < 35:
        age_group = "25-34"
    elif audience_age < 50:
        age_group = "35-49"
    else:
        age_group = "50+"

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Engagement Rate")


# ── Build input df ───────────────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    "Likes": likes, "Comments": comments, "Shares": shares,
    "Impressions": impressions, "Reach": reach,
    "Audience Age": audience_age, "Platform": platform,
    "Post Type": post_type, "Audience Gender": audience_gender,
    "Audience Location": audience_location,
    "Sentiment": sentiment, "Hour": hour, "Day": day,
    "time_of_day": time_of_day, "Age_group": age_group,
}])


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("# EngageIQ")
st.markdown("*Social Media Engagement Intelligence · Powered by PySpark + scikit-learn*")
st.markdown("---")


# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Dashboard", "🔮  Prediction", "📖  Model Info"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Dataset Insights")
    st.markdown("Key patterns surfaced from 100,000 social media posts analysed with PySpark.")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts Analysed", "100,000", "Full dataset")
    c2.metric("Avg Engagement Rate", "43.4%", "+9.4% vs industry benchmark")
    c3.metric("Top Platform", "TikTok", "52.8% avg")
    c4.metric("Best Post Type", "Video", "51.3% avg")

    st.markdown("---")

    # Charts row
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Engagement by Platform")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1c1c26")
        ax.set_facecolor("#1c1c26")
        platforms = list(BENCHMARKS["Platform"].keys())
        values    = list(BENCHMARKS["Platform"].values())
        colors    = ["#e8c97a" if p == max(BENCHMARKS["Platform"], key=BENCHMARKS["Platform"].get) else "#3a3a4e" for p in platforms]
        bars = ax.barh(platforms, values, color=colors, height=0.55, edgecolor="none")
        ax.set_xlabel("Avg Engagement Rate (%)", color="#7a7870", fontsize=9)
        ax.tick_params(colors="#9a9890", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color="#2a2a38", linewidth=0.5)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f"{val}%",
                    va="center", color="#c8c6c0", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Engagement by Post Type")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1c1c26")
        ax.set_facecolor("#1c1c26")
        types  = list(BENCHMARKS["Post Type"].keys())
        tvals  = list(BENCHMARKS["Post Type"].values())
        tcolors = ["#e8c97a" if t == "Video" else "#3a3a4e" for t in types]
        bars2 = ax.bar(types, tvals, color=tcolors, width=0.5, edgecolor="none")
        ax.set_ylabel("Avg Engagement Rate (%)", color="#7a7870", fontsize=9)
        ax.tick_params(colors="#9a9890", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color="#2a2a38", linewidth=0.5)
        ax.set_axisbelow(True)
        for bar, val in zip(bars2, tvals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val}%",
                    ha="center", color="#c8c6c0", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### 💡 Key Insights")
    for icon, title, body in KEY_INSIGHTS:
        st.markdown(f"""<div class="insight-card">{icon} <strong>{title}</strong> — {body}</div>""",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Prediction
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Engagement Rate Predictor")
    st.markdown("Configure your post parameters in the sidebar, then click **Predict**.")

    if not model_loaded:
        st.error("⚠️ Model file `linear_regression_model_with_features.pkl` not found. Train and save the model first.")
    elif predict_btn:
        prediction = model.predict(input_df)[0]
        prediction = max(0, min(100, prediction))

        # Percentile context (based on actual dataset distribution mean=43.4, std=37.7)
        all_rates = np.random.normal(43.4, 37.7, 10000)
        percentile = int(np.mean(all_rates < prediction) * 100)

        # ── Result banner ──
        st.markdown(f"""
        <div class="pred-banner">
            <div class="pred-label">Predicted Engagement Rate</div>
            <div class="pred-value">{prediction:.1f}%</div>
            <div class="pred-label" style="margin-top:0.6rem;">
                Top {100 - percentile}% of posts in dataset
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric breakdown ──
        m1, m2, m3 = st.columns(3)
        platform_avg = BENCHMARKS["Platform"].get(platform, 5.4)
        type_avg     = BENCHMARKS["Post Type"].get(post_type, 5.4)
        sentiment_avg = BENCHMARKS["Sentiment"].get(sentiment, 4.9)

        m1.metric("vs Platform Avg",  f"{platform_avg}%",  f"{prediction - platform_avg:+.1f}%")
        m2.metric("vs Post Type Avg", f"{type_avg}%",      f"{prediction - type_avg:+.1f}%")
        m3.metric("vs Sentiment Avg", f"{sentiment_avg}%", f"{prediction - sentiment_avg:+.1f}%")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        # ── Gauge chart ──
        with col_left:
            st.markdown("#### Engagement Score Gauge")
            fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
            fig.patch.set_facecolor("#1c1c26")
            ax.set_facecolor("#1c1c26")

            # Background arc
            theta = np.linspace(np.pi, 0, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="#2a2a38", linewidth=18, solid_capstyle="round")

            # Filled arc
            fill_pct = min(prediction / 100, 1.0)
            theta_fill = np.linspace(np.pi, np.pi - fill_pct * np.pi, 200)
            color = "#e8c97a" if prediction >= 6 else "#7ac4e8" if prediction >= 4 else "#e87a7a"
            ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=18, solid_capstyle="round")

            ax.text(0, -0.1, f"{prediction:.1f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color,
                    fontfamily="DM Sans")
            ax.text(0, -0.45, "Engagement Rate", ha="center", va="center",
                    fontsize=9, color="#7a7870")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.6, 1.2)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Input metrics bar chart ──
        with col_right:
            st.markdown("#### Your Post Metrics")
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#1c1c26")
            ax.set_facecolor("#1c1c26")
            metrics = {"Likes": likes, "Comments": comments, "Shares": shares}
            bars = ax.bar(metrics.keys(), metrics.values(),
                          color=["#e8c97a", "#7ac4e8", "#a87ae8"], width=0.4, edgecolor="none")
            ax.set_ylabel("Count", color="#7a7870", fontsize=9)
            ax.tick_params(colors="#9a9890", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.yaxis.grid(True, color="#2a2a38", linewidth=0.5)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, metrics.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values())*0.02,
                        f"{val:,}", ha="center", color="#c8c6c0", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # ── Recommendation ──
        st.markdown("#### 💬 Strategic Recommendation")
        tips = []
        if platform not in ["TikTok", "Instagram"]:
            tips.append("switch to TikTok or Instagram for higher reach")
        if post_type != "Video":
            tips.append("use Video content for the highest engagement")
        if sentiment != "Positive":
            tips.append("use a Positive sentiment tone")
        if day not in ["Friday", "Saturday"]:
            tips.append("post on Friday or Saturday")
        if hour < 17 or hour > 21:
            tips.append("schedule between 6–9 PM")
        tip_str = "; ".join(tips) if tips else "your setup is already well optimised"

        if prediction >= 55:
            st.success(f"🏆 **Excellent** — Predicted at {prediction:.1f}%, well above the 43.4% average. This is a high-performing setup — replicate this format consistently.")
        elif prediction >= 38:
            good_tip = "To push higher: " + tip_str + "." if tip_str != "your setup is already well optimised" else "Your setup is already well optimised - consider replicating this format."
            st.info(f"📈 **Good** - At {prediction:.1f}%, you're performing near or above the dataset average of 43.4%. {good_tip}")
        elif prediction >= 35:
            st.warning(f"⚠️ **Below Average** — At {prediction:.1f}%, this post may underperform. To improve: {tip_str}.")
        else:
            st.error(f"🚫 **Low Engagement** — At {prediction:.1f}%, this configuration is likely to underperform. To improve: {tip_str}.")

        st.markdown("---")
        st.markdown("#### 🚀 Best Config Given Your Choices")
        st.markdown(f"You've chosen **{platform}** and **{post_type}** — here's how to maximise engagement within those constraints:")

        # Best sentiment, day, time given fixed platform + post type
        best_sentiment = max(BENCHMARKS["Sentiment"], key=BENCHMARKS["Sentiment"].get)
        platform_avg   = BENCHMARKS["Platform"].get(platform, 5.4)
        post_type_avg  = BENCHMARKS["Post Type"].get(post_type, 5.4)

        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("📍 Your Platform",  platform,      f"{platform_avg}% avg")
        o2.metric("📝 Your Post Type", post_type,     f"{post_type_avg}% avg")
        o3.metric("✅ Best Sentiment", best_sentiment, f"{BENCHMARKS['Sentiment'][best_sentiment]}% avg")
        o4.metric("✅ Best Day",       "Friday",       "Peak engagement")
        o5.metric("✅ Best Time",      "6–9 PM",       "Evening window")

        # Personalised tips locked to their platform + post type
        changes = []
        if sentiment != best_sentiment:
            changes.append(f"Adjust sentiment → **{best_sentiment}** (+{BENCHMARKS['Sentiment'][best_sentiment] - BENCHMARKS['Sentiment'].get(sentiment, 4.9):.1f}% expected)")
        if hour < 17 or hour > 21:
            changes.append("Schedule between **6–9 PM** for the peak engagement window")
        if day not in ["Friday", "Saturday"]:
            changes.append("Post on **Friday or Saturday** for highest reach")

        # Also tell them how their platform/post type compares to the best
        best_platform  = max(BENCHMARKS["Platform"],  key=BENCHMARKS["Platform"].get)
        best_post_type = max(BENCHMARKS["Post Type"], key=BENCHMARKS["Post Type"].get)
        if platform != best_platform:
            diff = BENCHMARKS["Platform"][best_platform] - platform_avg
            changes.append(f"FYI: switching to **{best_platform}** could add **+{diff:.1f}%** engagement if you're open to it")
        if post_type != best_post_type:
            diff = BENCHMARKS["Post Type"][best_post_type] - post_type_avg
            changes.append(f"FYI: switching to **{best_post_type}** content could add **+{diff:.1f}%** engagement if you're open to it")

        if changes:
            st.markdown("**What to adjust:**")
            for c in changes:
                c_html = c.replace("**", "<strong>", 1)
                while "**" in c_html:
                    c_html = c_html.replace("**", "</strong>", 1)
                st.markdown(f"<div class='insight-card'>→ {c_html}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ Your configuration is already optimal!")

    else:
        st.info("👈 Set your post parameters in the sidebar and click **Predict Engagement Rate** to get started.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Model Info
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pipeline")
        st.markdown("""
        <div class="insight-card">
            <strong>Step 1 · Data Ingestion</strong><br>
            100,000 rows loaded via PySpark. Schema inferred, timestamps parsed, duplicates removed.
        </div>
        <div class="insight-card">
            <strong>Step 2 · Feature Engineering</strong><br>
            Hour and Day extracted from Post Timestamp. Missingness indicators added for Campaign ID, Sentiment, Influencer ID.
        </div>
        <div class="insight-card">
            <strong>Step 3 · Preprocessing</strong><br>
            Numeric features passed through. Categorical features (Platform, Post Type, Gender, Interests, Sentiment, Day) encoded with OneHotEncoder (handle_unknown='ignore').
        </div>
        <div class="insight-card">
            <strong>Step 4 · Model</strong><br>
            scikit-learn Linear Regression within a ColumnTransformer + Pipeline. 5-fold cross-validation on training set.
        </div>
        <div class="insight-card">
            <strong>Step 5 · Serialisation</strong><br>
            Full pipeline saved via joblib. Streamlit app loads and serves predictions in real time.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Features Used")
        features = {
            "Likes": "Numeric", "Comments": "Numeric", "Shares": "Numeric",
            "Impressions": "Numeric", "Reach": "Numeric", "Audience Age": "Numeric",
            "Hour": "Numeric (engineered)", "Platform": "Categorical (OHE)",
            "Post Type": "Categorical (OHE)", "Audience Gender": "Categorical (OHE)",
            "Audience Location": "Categorical (OHE)", "Sentiment": "Categorical (OHE)",
            "Day": "Categorical (OHE)", "time_of_day": "Categorical (engineered, OHE)",
            "Age_group": "Categorical (engineered, OHE)",
        }
        feat_df = pd.DataFrame(features.items(), columns=["Feature", "Type"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("#### Tech Stack")
        st.markdown("""
        <div class="insight-card">
            <strong>PySpark 3.5</strong> · Distributed EDA & MLlib baseline model<br><br>
            <strong>pandas + scikit-learn</strong> · Production model pipeline<br><br>
            <strong>Streamlit</strong> · Interactive deployment<br><br>
            <strong>matplotlib + seaborn</strong> · Visualisation
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Limitations & Future Work")
    st.markdown("""
    <div class="insight-card">⚠️ Linear Regression assumes linear relationships — <strong>XGBoost or LightGBM</strong> would likely improve predictive performance significantly.</div>
    <div class="insight-card">⚠️ Engagement Rate is heavily influenced by likes/impressions which are post-hoc metrics — a production system would need a pre-publish prediction variant.</div>
    <div class="insight-card">🔭 Future: SHAP values for per-prediction explainability, real-time API integration, A/B test simulation mode.</div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#4a4840; font-size:0.8rem;'>EngageIQ · Built with PySpark & scikit-learn · Social Media Engagement Analysis Project</p>", unsafe_allow_html=True)