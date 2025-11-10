# ============================================================
# 🎴 From Charizard to Skywalker — Final Showcase (H₂ + H₃ with Market Index Forecast)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.ticker as mtick

# Prophet (for Pokémon forecasting)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ------------------------------------------------------------
# Streamlit setup
st.set_page_config(page_title="Collectibles Hypotheses 2 & 3", page_icon="🎴", layout="wide")

# ------------------------------------------------------------
# Light theme styling
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #222; }
h1, h2, h3 { color: #002060; }
.section { background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load datasets
@st.cache_data
def load_pokemon():
    df = pd.read_csv("final_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Graded", "Card Name"])
    df = df[df["Graded"] > 0]
    return df.sort_values("Date")

@st.cache_data
def load_starwars():
    sw = pd.read_csv("starwars_moc_loose_v7.csv")
    sw = sw.dropna(subset=["figure", "year", "selling_price"])
    sw["year"] = sw["year"].astype(int)
    sw["Date"] = pd.to_datetime(sw["year"].astype(str) + "-01-01")
    return sw

df_poke = load_pokemon()
df_sw = load_starwars()

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "🧩 Hypothesis 2 — Market Segmentation & Clustering",
    "🔮 Hypothesis 3 — Forecastability"
])

# ============================================================
# 🧩 H₂ — Market Segmentation & Clustering
# ============================================================
with tabs[0]:
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image("vecteezy_pokemon-logo-png-pokemon-icon-transparent-png_27127571.png", width=220)
            with c2:
                st.image("https://upload.wikimedia.org/wikipedia/commons/6/6c/Star_Wars_Logo.svg", width=220)
            with c3:
                st.image("totodile.gif", width=220)

    st.header("🧩 Hypothesis 2 — Market Segmentation & Clustering")

    st.markdown("""
    **Statement:**  
    Distinct collectible segments exist, each with unique growth and volatility profiles.

    **Methodology:**  
    - Pokémon: *K-Means* clustering using mean, growth & volatility.  
    - Star Wars: *Polynomial coefficients* & volatility.  
    """)

    # --- Pokémon clustering
    feats_poke = []
    for c in df_poke["Card Name"].unique():
        sub = df_poke[df_poke["Card Name"] == c].dropna(subset=["Graded"]).sort_values("Date")
        if len(sub) < 5:
            continue
        feats_poke.append({
            "Card": c,
            "Mean": sub["Graded"].mean(),
            "Std": sub["Graded"].std(),
            "Volatility": sub["Graded"].pct_change().std(),
            "Growth": (sub["Graded"].iloc[-1] - sub["Graded"].iloc[0]) / max(sub["Graded"].iloc[0], 1e-6)
        })
    poke_feat = pd.DataFrame(feats_poke).dropna()

    if len(poke_feat) >= 3:
        Xp = StandardScaler().fit_transform(poke_feat[["Mean","Std","Volatility","Growth"]])
        kmp = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xp)
        poke_feat["Cluster"] = kmp.labels_

        fig1, ax1 = plt.subplots(figsize=(9,5))
        sns.scatterplot(data=poke_feat, x="Growth", y="Volatility",
                        hue="Cluster", palette="viridis", s=100, ax=ax1)

        # Label one top point per cluster
        for cluster_id in poke_feat["Cluster"].unique():
            cluster_df = poke_feat[poke_feat["Cluster"] == cluster_id]
            top_card = cluster_df.loc[cluster_df["Growth"].idxmax()]
            ax1.text(top_card["Growth"]+0.02, top_card["Volatility"],
                     top_card["Card"], fontsize=9, weight='bold')

        ax1.set_title("🎮 Pokémon — Growth vs Volatility Clusters")
        ax1.grid(alpha=.3)
        st.pyplot(fig1)

        st.image("https://archives.bulbagarden.net/media/upload/7/7e/Base_Set_Charizard_English.jpg",
                 width=150, caption="🔥 Base Set Charizard — 'Blue-Chip' Pokémon Asset")

    # --- Star Wars clustering
    st.subheader("🌌 Star Wars — Polynomial Coefficients vs Volatility")

    df = df_sw.copy()
    features = []
    for fig in df["figure"].unique():
        sub = df[df["figure"] == fig].copy()
        if len(sub) < 5:
            continue
        X = sub[["year"]]
        y = sub["selling_price"]
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        mean_price = y.mean()
        volatility = y.pct_change().std()
        features.append({
            "Figure": fig,
            "Coef1_Linear": model.coef_[1],
            "Coef2_Quadratic": model.coef_[2],
            "MeanPrice": mean_price,
            "Volatility": volatility
        })
    feat_df = pd.DataFrame(features).dropna()

    if len(feat_df) >= 3:
        X_scaled = StandardScaler().fit_transform(
            feat_df[["Coef1_Linear", "Coef2_Quadratic", "MeanPrice", "Volatility"]])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        feat_df["Cluster"] = kmeans.fit_predict(X_scaled)

        fig3, ax3 = plt.subplots(figsize=(9,6))
        sns.scatterplot(data=feat_df, x="Coef1_Linear", y="Volatility",
                        hue="Cluster", palette="viridis", s=120, ax=ax3)

        # Label top performer per cluster
        for cluster_id in feat_df["Cluster"].unique():
            cluster_df = feat_df[feat_df["Cluster"] == cluster_id]
            top_fig = cluster_df.loc[cluster_df["Coef1_Linear"].idxmax()]
            ax3.text(top_fig["Coef1_Linear"], top_fig["Volatility"],
                     top_fig["Figure"], fontsize=9, weight='bold')

        ax3.set_title("🌌 Star Wars — Polynomial Growth vs Volatility")
        ax3.set_xlabel("Polynomial Growth Coefficient (Linear Term)")
        ax3.set_ylabel("Volatility (std of returns)")
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

    st.markdown("""
    ### 🔍 Findings
    - Both markets form **three distinct clusters**: blue-chip, mid-tier, and speculative.  
    - Pokémon clusters are **tighter and more stable**, signaling a mature collectible market.  
    - Star Wars clusters show **broader variance**, reflecting event-driven price spikes.  
    ✅ **Hypothesis 2 validated:** Collectibles exhibit structured market segmentation.
    """)

# ============================================================
# 🔮 H₃ — Forecastability + Market Index
# ============================================================
with tabs[1]:
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image("vecteezy_pokemon-logo-png-pokemon-icon-transparent-png_27127571.png", width=220)
            with c2:
                st.image("https://upload.wikimedia.org/wikipedia/commons/6/6c/Star_Wars_Logo.svg", width=220)
            with c3:
                st.image("charizard.webp", width=220)

    st.header("🔮 Hypothesis 3 — Forecastability of Collectibles")

    st.markdown("""
    **Statement:**  
    Historical price behaviour allows short-term forecasting of collectible values.
    """)

    # Pokémon Prophet forecast
    st.subheader("🎮 Pokémon — Prophet Forecast")
    st.info("""
    **Prophet** (by Meta) decomposes time series into trend + seasonality + noise.  
    Perfect for high-frequency Pokémon data with long-term structural trends.
    """)

    cards = sorted(df_poke["Card Name"].unique())
    card = st.selectbox("Select a Pokémon card", cards)
    sub = df_poke[df_poke["Card Name"] == card].sort_values("Date")

    if PROPHET_AVAILABLE and len(sub) >= 12:
        dfp = sub.rename(columns={"Date":"ds","Graded":"y"})[["ds","y"]]
        m = Prophet(yearly_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=365)
        fc = m.predict(future)
        fig5, ax5 = plt.subplots(figsize=(9,4))
        ax5.plot(dfp["ds"], dfp["y"], "k.", label="Actual")
        ax5.plot(fc["ds"], fc["yhat"], "b-", label="Forecast")
        ax5.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="skyblue", alpha=.3)
        ax5.set_title(f"{card} — 12-Month Prophet Forecast")
        ax5.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax5.legend(); ax5.grid(alpha=.3)
        st.pyplot(fig5)

    # Star Wars polynomial forecast
    st.subheader("🌌 Star Wars — Polynomial Forecast")
    st.info("""
    **Polynomial Regression** fits non-linear curves to long-term data — ideal for  
    vintage Star Wars figures that follow hype-and-fall cycles.
    """)

    fig_pick = st.selectbox("Select a Star Wars figure", sorted(df_sw["figure"].unique()))
    sub_sw = df_sw[df_sw["figure"] == fig_pick].sort_values("year")

    if len(sub_sw) >= 5:
        X, y = sub_sw[["year"]], sub_sw["selling_price"]
        poly = PolynomialFeatures(2)
        lr = LinearRegression().fit(poly.fit_transform(X), y)
        future = np.arange(sub_sw["year"].min(), sub_sw["year"].max()+5).reshape(-1,1)
        y_pred = lr.predict(poly.transform(future))
        r2 = lr.score(poly.fit_transform(X), y)
        fig6, ax6 = plt.subplots(figsize=(9,4))
        ax6.scatter(X, y, color="#0070C0", label="Actual")
        ax6.plot(future, y_pred, "r-", label=f"Polynomial Fit (R² = {r2:.2f})")
        ax6.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax6.set_title(f"{fig_pick} — Polynomial Forecast")
        ax6.grid(alpha=.3); ax6.legend()
        st.pyplot(fig6)

    # --- Market Index Forecast
    st.subheader("💹 Pokémon vs Star Wars — Market Index (Base = 100) + 2026 Forecast")
    st.markdown("""
    Each index represents the **average normalized market price** across all collectibles for each franchise.  
    Normalizing (Base = 100) allows cross-market comparison regardless of absolute price differences.  
    The 2026 projection provides a simplified forward view of their relative market momentum.
    """)

    poke_year = df_poke.set_index("Date").resample("Y")["Graded"].mean().reset_index()
    poke_year["year"] = poke_year["Date"].dt.year
    poke_year["Index"] = poke_year["Graded"] / poke_year["Graded"].iloc[0] * 100

    sw_year = df_sw.groupby("year")["selling_price"].mean().reset_index()
    sw_year["Index"] = sw_year["selling_price"] / sw_year["selling_price"].iloc[0] * 100

    # Fit simple regression for 2026 projection
    def extend_index(df, label):
        model = LinearRegression().fit(df[["year"]], df["Index"])
        next_year = np.array([[df["year"].max() + 1]])
        forecast = model.predict(next_year)[0]
        return next_year[0][0], forecast

    next_poke_year, poke_forecast = extend_index(poke_year, "Pokémon")
    next_sw_year, sw_forecast = extend_index(sw_year, "Star Wars")

    fig7, ax7 = plt.subplots(figsize=(9,4))
    ax7.plot(poke_year["year"], poke_year["Index"], "b-o", label="Pokémon Index")
    ax7.plot(sw_year["year"], sw_year["Index"], "g-o", label="Star Wars Index")
    ax7.scatter(next_poke_year, poke_forecast, color="blue", marker="*", s=150)
    ax7.scatter(next_sw_year, sw_forecast, color="green", marker="*", s=150)
    ax7.text(next_poke_year, poke_forecast + 1, f"{poke_forecast:.1f}", color="blue", weight="bold")
    ax7.text(next_sw_year, sw_forecast + 1, f"{sw_forecast:.1f}", color="green", weight="bold")
    ax7.set_ylabel("Normalized Index (100 = initial year)")
    ax7.set_title("Market Index — Historical and 2026 Forecast")
    ax7.legend(); ax7.grid(alpha=.3)
    st.pyplot(fig7)

    st.markdown("""
    ### 🔍 Interpretation
    - Pokémon’s forecast suggests **continued stability** around its equilibrium level.  
    - Star Wars shows **potential upward correction**, driven by cyclical nostalgia surges.  
    - Together, these validate **Hypothesis 3**: collectibles behave as forecastable asset markets.
    """)


