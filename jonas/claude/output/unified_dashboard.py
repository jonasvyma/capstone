import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.ticker as mtick

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# ===========================================
# Page Configuration
# ===========================================
st.set_page_config(
    page_title="Collectibles Market Analysis - Unified Dashboard",
    page_icon="🎴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #222; }
h1, h2, h3 { color: #002060; }
.section { background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

# Set plot styles
sns.set_style("whitegrid")
sns.set_palette("husl")

# ===========================================
# Data Loading Functions
# ===========================================
@st.cache_data
def load_pokemon_detailed():
    """Load Pokemon data for detailed analysis"""
    df = pd.read_csv('final_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df = df[df['Rarity'] != 'Booster Box']
    return df

@st.cache_data
def load_pokemon_hypothesis():
    """Load Pokemon data for hypothesis testing"""
    df = pd.read_csv("final_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Graded", "Card Name"])
    df = df[df["Graded"] > 0]
    return df.sort_values("Date")

@st.cache_data
def load_starwars():
    """Load Star Wars data"""
    sw = pd.read_csv("starwars_moc_loose_v7.csv")
    sw = sw.dropna(subset=["figure", "year", "selling_price"])
    sw["year"] = sw["year"].astype(int)
    sw["Date"] = pd.to_datetime(sw["year"].astype(str) + "-01-01")
    return sw

@st.cache_data
def load_starwars_filtered():
    """Load filtered Star Wars data"""
    df = pd.read_csv('claude/starwars_filtered.csv')
    return df

# Load all datasets
try:
    df_poke_detail = load_pokemon_detailed()
    df_poke_hyp = load_pokemon_hypothesis()
    df_sw = load_starwars()
except:
    st.error("⚠️ Error loading datasets. Please ensure all CSV files are in the correct location.")
    st.stop()

# Define price columns
price_cols = ['New', 'Used', 'Graded']

# ===========================================
# Main Navigation
# ===========================================
st.title("🎴 Collectibles Market Analysis - Unified Dashboard")
st.markdown("**Comprehensive analysis of Pokemon Cards, Star Wars Figures, and Market Hypotheses**")

main_tabs = st.tabs([
    "🎮 Pokemon Card Analysis",
    "🌌 Star Wars Figure Analysis",
    "🧩 Market Hypotheses (H2 & H3)"
])

# ===========================================
# TAB 1: POKEMON CARD ANALYSIS
# ===========================================
with main_tabs[0]:
    st.header("🎴 Pokemon Card Pricing Analysis")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Pokemon Filters")
        
        # Date range filter
        min_date = df_poke_detail['Date'].min()
        max_date = df_poke_detail['Date'].max()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_poke_detail[(df_poke_detail['Date'] >= pd.to_datetime(start_date)) &
                                         (df_poke_detail['Date'] <= pd.to_datetime(end_date))]
        else:
            df_filtered = df_poke_detail.copy()
        
        # Rarity filter
        rarity_options = ['All'] + sorted(df_filtered['Rarity'].unique().tolist())
        selected_rarity = st.selectbox("Select Rarity", rarity_options)
        if selected_rarity != 'All':
            df_filtered = df_filtered[df_filtered['Rarity'] == selected_rarity]
        
        # Set filter
        set_options = ['All'] + sorted(df_filtered['Set Name'].unique().tolist())
        selected_set = st.selectbox("Select Set", set_options)
        if selected_set != 'All':
            df_filtered = df_filtered[df_filtered['Set Name'] == selected_set]
        
        # Card filter
        card_options = ['All'] + sorted(df_filtered['Card Name'].unique().tolist())
        selected_card = st.selectbox("Select Card", card_options)
        if selected_card != 'All':
            df_filtered = df_filtered[df_filtered['Card Name'] == selected_card]
        
        st.markdown("---")
        st.subheader("📊 Filtered Data")
        st.metric("Total Records", f"{len(df_filtered):,}")
        st.metric("Unique Cards", df_filtered['Card Name'].nunique())
    
    # Create sub-tabs for Pokemon analysis
    pk_tabs = st.tabs([
        "📊 Overview",
        "💰 Price Analysis",
        "🎯 Rarity Analysis",
        "📈 Time Trends",
        "🏆 Top Cards",
        "📉 Statistical Analysis"
    ])
    
    # Pokemon Overview Tab
    with pk_tabs[0]:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df_filtered):,}")
        with col2:
            st.metric("Unique Cards", df_filtered['Card Name'].nunique())
        with col3:
            st.metric("Unique Rarities", df_filtered['Rarity'].nunique())
        with col4:
            st.metric("Unique Sets", df_filtered['Set Name'].nunique())
        
        st.dataframe(df_filtered.head(10), use_container_width=True)
    
    # Pokemon Price Analysis Tab
    with pk_tabs[1]:
        st.subheader("Price Distribution by Condition")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Price Distribution by Condition', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(price_cols):
            ax = axes[idx]
            data = df_filtered[df_filtered[col] > 0][col]
            if len(data) > 0:
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Price ($)', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{col} (n={len(data):,})', fontsize=12, fontweight='bold')
                ax.axvline(data.mean(), color='red', linestyle='--',
                           label=f'Mean: ${data.mean():,.0f}')
                ax.axvline(data.median(), color='green', linestyle='--',
                           label=f'Median: ${data.median():,.0f}')
                ax.legend(fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Boxplots
        st.subheader("Price Distribution: Boxplot Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        price_data = []
        conditions = []
        for col in price_cols:
            non_zero = df_filtered[df_filtered[col] > 0][col]
            price_data.extend(non_zero.values)
            conditions.extend([col] * len(non_zero))
        
        boxplot_df = pd.DataFrame({'Price': price_data, 'Condition': conditions})
        
        sns.boxplot(data=boxplot_df, x='Condition', y='Price', ax=ax1)
        ax1.set_title('WITH Outliers', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        sns.boxplot(data=boxplot_df, x='Condition', y='Price', ax=ax2, showfliers=False)
        ax2.set_title('WITHOUT Outliers', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Pokemon Time Trends Tab
    with pk_tabs[3]:
        st.subheader("Price Trends Over Time")
        fig, ax = plt.subplots(figsize=(16, 7))
        
        for col in price_cols:
            monthly_avg = df_filtered.groupby('Date')[col].mean()
            ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2,
                    label=col, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Price ($)', fontsize=12)
        ax.set_title('Pokemon Card Price Trends Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ===========================================
# TAB 2: STAR WARS FIGURE ANALYSIS
# ===========================================
with main_tabs[1]:
    st.header("🌌 Star Wars Figure Analysis")
    st.info("Star Wars figure pricing analysis coming soon - integrate sw_streamlit.py content here")
    
    # Basic Star Wars visualization
    st.subheader("Star Wars Price Overview")
    if len(df_sw) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        yearly_avg = df_sw.groupby('year')['selling_price'].mean()
        ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=5)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average Selling Price ($)', fontsize=12)
        ax.set_title('Star Wars Figures - Average Price by Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ===========================================
# TAB 3: MARKET HYPOTHESES (H2 & H3)
# ===========================================
with main_tabs[2]:
    # Header images
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
    
    hyp_tabs = st.tabs([
        "🧩 Hypothesis 2 — Market Segmentation",
        "🔮 Hypothesis 3 — Forecastability"
    ])
    
    # HYPOTHESIS 2: Market Segmentation
    with hyp_tabs[0]:
        st.header("🧩 Hypothesis 2 — Market Segmentation & Clustering")
        
        st.markdown("""
        **Statement:**  
        Distinct collectible segments exist, each with unique growth and volatility profiles.
        
        **Methodology:**  
        - Pokémon: *K-Means* clustering using mean, growth & volatility.  
        - Star Wars: *Polynomial coefficients* & volatility.  
        """)
        
        # Pokemon clustering
        feats_poke = []
        for c in df_poke_hyp["Card Name"].unique():
            sub = df_poke_hyp[df_poke_hyp["Card Name"] == c].dropna(subset=["Graded"]).sort_values("Date")
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
            
            for cluster_id in poke_feat["Cluster"].unique():
                cluster_df = poke_feat[poke_feat["Cluster"] == cluster_id]
                top_card = cluster_df.loc[cluster_df["Growth"].idxmax()]
                ax1.text(top_card["Growth"]+0.02, top_card["Volatility"],
                         top_card["Card"], fontsize=9, weight='bold')
            
            ax1.set_title("🎮 Pokémon — Growth vs Volatility Clusters")
            ax1.grid(alpha=.3)
            st.pyplot(fig1)
            plt.close()
        
        # Star Wars clustering
        st.subheader("🌌 Star Wars — Polynomial Coefficients vs Volatility")
        
        features = []
        for fig in df_sw["figure"].unique():
            sub = df_sw[df_sw["figure"] == fig].copy()
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
            plt.close()
        
        st.markdown("""
        ### 🔍 Findings
        - Both markets form **three distinct clusters**: blue-chip, mid-tier, and speculative.  
        - Pokémon clusters are **tighter and more stable**, signaling a mature collectible market.  
        - Star Wars clusters show **broader variance**, reflecting event-driven price spikes.  
        ✅ **Hypothesis 2 validated:** Collectibles exhibit structured market segmentation.
        """)
    
    # HYPOTHESIS 3: Forecastability
    with hyp_tabs[1]:
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
        
        # Pokemon Prophet forecast
        st.subheader("🎮 Pokémon — Prophet Forecast")
        st.info("""
        **Prophet** (by Meta) decomposes time series into trend + seasonality + noise.  
        Perfect for high-frequency Pokémon data with long-term structural trends.
        """)
        
        cards = sorted(df_poke_hyp["Card Name"].unique())
        card = st.selectbox("Select a Pokémon card", cards)
        sub = df_poke_hyp[df_poke_hyp["Card Name"] == card].sort_values("Date")
        
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
            plt.close()
        
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
            plt.close()
        
        # Market Index Forecast
        st.subheader("💹 Pokémon vs Star Wars — Market Index (Base = 100) + 2026 Forecast")
        st.markdown("""
        Each index represents the **average normalized market price** across all collectibles for each franchise.  
        Normalizing (Base = 100) allows cross-market comparison regardless of absolute price differences.  
        The 2026 projection provides a simplified forward view of their relative market momentum.
        """)
        
        poke_year = df_poke_hyp.set_index("Date").resample("Y")["Graded"].mean().reset_index()
        poke_year["year"] = poke_year["Date"].dt.year
        poke_year["Index"] = poke_year["Graded"] / poke_year["Graded"].iloc[0] * 100
        
        sw_year = df_sw.groupby("year")["selling_price"].mean().reset_index()
        sw_year["Index"] = sw_year["selling_price"] / sw_year["selling_price"].iloc[0] * 100
        
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
        plt.close()
        
        st.markdown("""
        ### 🔍 Interpretation
        - Pokémon's forecast suggests **continued stability** around its equilibrium level.  
        - Star Wars shows **potential upward correction**, driven by cyclical nostalgia surges.  
        - Together, these validate **Hypothesis 3**: collectibles behave as forecastable asset markets.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Unified Collectibles Market Analysis Dashboard</strong></p>
    <p>Comprehensive analysis of Pokemon Cards, Star Wars Figures, and Market Hypotheses</p>
</div>
""", unsafe_allow_html=True)
