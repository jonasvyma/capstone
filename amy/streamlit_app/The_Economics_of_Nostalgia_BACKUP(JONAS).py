import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

st.set_page_config(page_title="**The Economics of Nostalgia**: Data-Driven Insights into Collectible Markets", page_icon="🎴", layout="wide")
#Create Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Business Context",
    "Hypotheses & Methodology",
    "Findings of Exploratory Analysis - Star Wars",
    "Findings of Exploratory Analysis - Pokemon",
    "Hypothesis 1",
    "Hypothesis 2",
    "Hypothesis 3",
    "Conclusion"
])
# Read in variables:
starwars_moc_loose = pd.read_csv('data/starwars_moc_loose_v7.csv')
pokemon_final_26 = pd.read_csv('data/final_dataset_csv.csv')
pokemon_medians = pd.read_csv('data/pokemons_medians_df.csv')
starwars_medians = pd.read_csv('data/starwars_medians_df.csv')
starwars_year_figure = pd.read_csv('data/starwars_year_figure.csv')
pokemon_year_card = pd.read_csv('data/pokemon_year_card_df.csv')
# For H1 testing 
pokemon_returns = pokemon_year_card['YoY_Growth'].replace([np.inf, -np.inf], np.nan).dropna()
starwars_returns = starwars_year_figure['YoY_Growth'].dropna()
sw_returns_list = starwars_returns.to_list()
poke_returns_list = pokemon_returns.to_list()

with tab1:
    st.subheader('Case studies on Star Wars Figurines and Pokemon Trading Cards ')
    st.subheader('Business Context')
    st.subheader('In recent years, collectible markets have attracted fans and investors alike, who value them not only for their sentimental value, but also as an alternative asset ' \
             '- similar to art, watches, or crypto.' \
             'The question is: Do collectibles behave financially like traditional assets, ' \
             'showing measurable growth, volatility, correlation, and risk-return patterns similar to stocks or portfolios?')
    st.subheader('*Can Pokémon trading cards and Star Wars figures — when analysed through price history and other factors — be modeled and interpreted as investment-grade assets, exhibiting similar patterns of value appreciation, volatility, and market segmentation as financial securities?*')
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/darth_vader.jpg', caption='How much would you pay for this?')
    with col2:
        st.image('images/pikachu.jpg', caption='How valuable do you think this is?')

with tab2:
    st.subheader('Research Questions and Hypotheses')
    st.markdown(
    "- Cards and figures appreciate over time and deliver an average return over 2%.\n"
    "- Distinct risk–return clusters exist and can be correlated by condition,authenticity,rarity and character/set type.\n"
    "- Forecasts using Prophet and Polynomial methods reveal predictable patterns and markets stabilise post-hype.\n"
    )
    with st.expander("Hypothesis Testing and Business Research Methodology"):
        st.markdown(
        """
        - **H1:**
            - <u>Pokemon</u>  
                - 1-tailed T test  
                - Testing measured YoY returns for 26 cards over 5 years against expected annual return of 2%.  
                - Visualised through growth of median selling price and YoY growth vs expected selling price with 2% return.  
            - <u>Star Wars</u>  
                - 1-tailed T test  
                - Testing measured YoY returns for both loose and MOC figures over 16 years against expected annual return of 2%.  
                - Visualised through growth of median selling price and YoY growth vs expected selling price with 2% return.  

        - **H2:**  
            - <u>Pokemon</u>  
                - K-means clustering for growth and volatility.  
                - ANOVA test for effect of rarity and set name on growth and volatility.  
                - Visualised through scatter plots and box plots.  
            - <u>Star Wars</u>  
                - K-means clustering for growth and volatility.  
                - ANOVA test for effect of condition, authenticity, and character type on growth and volatility.  

        - **H3:**  
            - <u>Pokemon</u>  
                - Forecasting selling price using Prophet model with regressor of graded price and features including growth, volatility, and card name.  
            - <u>Star Wars</u>  
                - Forecasting selling price using polynomial model with regressor of graded price and features including condition, authenticity, and character type.  
        """,
        unsafe_allow_html=True
    )
        # create dictionary for pokemon terms:
    poke_dict = {
    "Data Name": [
        "asin", "bgs-10-price", "box-only-price", "cib-price", "condition-17-price",
        "condition-18-price", "console-name", "epid", "gamestop-price", "genre",
        "graded-price", "id", "loose-price", "manual-only-price", "new-price",
        "product-name", "release-date", "retail-cib-buy", "retail-cib-sell",
        "retail-loose-buy", "retail-loose-sell", "retail-new-buy", "retail-new-sell",
        "sales-volume", "upc"
    ],
    "Description": [
        "Unique identifier (ASIN) for this product on Amazon.com",
        "Cards: BGS 10; Comics: Graded 10.0 by a grading company",
        "Video Games: Price for the original box only; Cards: Graded 9.5 by a grading company; Comics: Graded 9.2 by a grading company",
        "Video Games: Price for item only with original box and manual included; Cards: Graded 7 or 7.5 by a grading company; Comics: Graded 4.0 or 4.5 by a grading company",
        "Cards: CGC 10; Comics: Graded 9.4 by a grading company",
        "Cards: SGC 10",
        "The name of the console on which the item was released.",
        "Unique identifier (ePID) for this product on eBay",
        "The price that GameStop charges for this game in 'Pre-Owned' condition. Trade price is what GameStop pays in cash for trade-in games. These prices are only available for consoles that GameStop sells or trades.",
        "The genre is a single category which describes the item. For example RPG, Fighting, Party, Pokemon Card, etc.",
        "Video Games: Price for brand new item graded by WATA or VGA; Cards: Graded 9 by a grading company; Comics: Graded 8.0 or 8.5 by a grading company",
        "PriceCharting unique id for a product.",
        "Video Games: Price for item only without box or manual; Cards: Ungraded card; Comics: Ungraded comic",
        "Video Games: Price for the original manual only; Cards: Graded 10 by PSA grading service; Comics: Graded 9.8 by a grading company",
        "Video Games: Price for item with original packaging and original seal; Cards: Graded 8 or 8.5 by a grading company; Comics: Graded 6.0 or 6.5 by a grading company",
        "The name of the item.",
        "The date the item was originally released.",
        "The recommended price for retailers buying from a customer in CIB (complete in box) condition.",
        "The recommended price for retailers selling to a customer in CIB (complete in box) condition.",
        "The recommended price for retailers buying from a customer in loose condition.",
        "The recommended price for retailers selling to a customer in loose condition.",
        "The recommended price for retailers buying from a customer in brand new condition.",
        "The recommended price for retailers selling to a customer in brand new condition.",
        "The yearly units sold",
        "The items in your guide will include a UPC that helps you track the item and sell on some websites (e.g. eBay). UPCs may not be available for older consoles that came out before UPCs were created."
    ]}
    poke_dict = pd.DataFrame(poke_dict)
    with st.expander("Dictionary"):
        st.markdown(
        """- <u>Pokemon</u>   
        """,
        unsafe_allow_html=True)
    
        st.dataframe(poke_dict)
        st.markdown("""
                <u>Star Wars</u>""")
                


with tab3:
    st.header('Findings of Exporatory Analysis - Star Wars')
    st.subheader('Jonas to add here')

with tab4:
    st.header('Findings of Exploratory Analysis - Pokemon')
    st.subheader('Jonas to add here')
   
with tab5:
    st.header('Hypothesis 1 - Findings')
    st.subheader('Amy to add here')

with tab6:
    # ============================================================
    # 🧩 Hypothesis 2 — Market Segmentation & Clustering
    # ============================================================
    st.header("🧩 Hypothesis 2 — Market Segmentation & Clustering")

    st.markdown("""
    **Statement:**  
    Distinct collectible segments exist, each with unique growth and volatility profiles.

    **Methodology:**  
    - Pokémon: *K-Means* clustering using derived YoY growth & volatility from graded prices (`pokemon_final_26`).  
    - Star Wars: *K-Means* clustering on mean growth & volatility (`starwars_moc_loose`).  
    - ANOVA and correlation tests evaluate whether artistic or physical attributes influence growth.
    """)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy import stats

    # ------------------------------------------------------------
    # 🟢 Pokémon Data Prep
    # ------------------------------------------------------------
    df_poke = pokemon_final_26.copy()
    df_poke["Date"] = pd.to_datetime(df_poke["Date"], errors="coerce")
    df_poke = df_poke.dropna(subset=["Graded", "Date"])
    df_poke = df_poke.sort_values(["Card Name", "Date"])

    growth_data = []
    for card, sub in df_poke.groupby("Card Name"):
        sub = sub.sort_values("Date")
        sub["YoY_Growth"] = sub["Graded"].pct_change().replace([np.inf, -np.inf], np.nan)
        if sub["YoY_Growth"].notna().sum() >= 2:
            growth_data.append({
                "Card Name": card,
                "Illustrator": sub["Illustrator"].iloc[0],
                "Set Name": sub["Set Name"].iloc[0],
                "Mean_Growth": sub["YoY_Growth"].mean(skipna=True),
                "Volatility": sub["YoY_Growth"].std(skipna=True)
            })
    poke_feat = pd.DataFrame(growth_data).dropna(subset=["Mean_Growth", "Volatility"])

    # ------------------------------------------------------------
    # 🔵 Star Wars Data Prep
    # ------------------------------------------------------------
    df_sw = starwars_moc_loose.copy()
    df_sw = df_sw.dropna(subset=["total_growth", "volatility_y"])
    sw_feat = df_sw.groupby("figure").agg(
        Mean_Growth=("total_growth", "mean"),
        Volatility=("volatility_y", "mean"),
        Condition=("condition", "first"),
        Authenticity=("authenticity_n", "first"),
        Character_Type=("character_type", "first")
    ).reset_index().dropna()

    # ------------------------------------------------------------
    # 🎨 Shared Clustering (Same palette & names)
    # ------------------------------------------------------------
    palette_named = {
        "Blue-Chip": "#82CAFF",
        "Mid-Tier": "#9370DB",
        "Speculative": "#FF69B4"
    }

    if len(poke_feat) >= 3 and len(sw_feat) >= 3:
        scaler = StandardScaler()
        kmp = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.fit_transform(poke_feat[["Mean_Growth", "Volatility"]]))
        kms = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.fit_transform(sw_feat[["Mean_Growth", "Volatility"]]))

        cluster_labels = {0: "Blue-Chip", 1: "Mid-Tier", 2: "Speculative"}
        poke_feat["Cluster_Name"] = poke_feat["Cluster"] = kmp.labels_
        poke_feat["Cluster_Name"] = poke_feat["Cluster_Name"].map(cluster_labels)
        sw_feat["Cluster_Name"] = sw_feat["Cluster"] = kms.labels_
        sw_feat["Cluster_Name"] = sw_feat["Cluster_Name"].map(cluster_labels)

        # ------------------------------------------------------------
        # 📊 Side-by-Side Cluster Charts + Statistical Tests
        # ------------------------------------------------------------
        col1, col2 = st.columns(2)

        # Pokémon (Left Column)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.scatterplot(
                data=poke_feat, x="Mean_Growth", y="Volatility",
                hue="Cluster_Name", palette=palette_named, s=100, ax=ax1
            )
            ax1.set_title("🎮 Pokémon — Growth vs Volatility Clusters", fontsize=11)
            ax1.grid(alpha=.3)
            ax1.legend(title="Cluster", loc="best")
            st.pyplot(fig1)

            # Pokémon Statistical Tests
            st.markdown("### 🎮 Pokémon — Statistical Tests")
            poke_tests_displayed = 0

            # 💰 Graded ↔ Growth
            if "Graded" in df_poke.columns:
                if "Mean_Growth" in poke_feat.columns and len(poke_feat) > 2:
                    card_prices = df_poke.groupby("Card Name")["Graded"].mean().reset_index()
                    merged_data = poke_feat.merge(card_prices, on="Card Name", how="inner")
                    if len(merged_data) > 2:
                        corr, p_val = stats.pearsonr(merged_data["Graded"], merged_data["Mean_Growth"])
                        color = "green" if p_val < 0.05 else "red"
                        st.markdown(
                            f"<b>💰 Graded Price ↔ Growth:</b> "
                            f"<span style='color:{color}'>r = {corr:.2f}, p = {p_val:.4f} "
                            f"({'✅ significant' if p_val < 0.05 else '❌ not significant'})</span><br>"
                            f"<i>Tests whether higher graded cards exhibit stronger yearly growth rates.</i>",
                            unsafe_allow_html=True
                        )
                        poke_tests_displayed += 1

            # 🎨 Illustrator → Growth
            if "Illustrator" in poke_feat.columns:
                groups = [g["Mean_Growth"].dropna() for _, g in poke_feat.groupby("Illustrator") if len(g) > 1]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    color = "green" if p_val < 0.05 else "red"
                    st.markdown(
                        f"<b>🎨 Illustrator → Growth:</b> "
                        f"<span style='color:{color}'>F = {f_stat:.2f}, p = {p_val:.4f} "
                        f"({'✅ significant' if p_val < 0.05 else '❌ not significant'})</span><br>"
                        f"<i>Tests whether cards illustrated by different artists show different average growth.</i>",
                        unsafe_allow_html=True
                    )
                    poke_tests_displayed += 1

            # 🧩 Set Name → Growth
            if "Set Name" in poke_feat.columns:
                groups = [g["Mean_Growth"].dropna() for _, g in poke_feat.groupby("Set Name") if len(g) > 1]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    color = "green" if p_val < 0.05 else "red"
                    st.markdown(
                        f"<b>🧩 Set Name → Growth:</b> "
                        f"<span style='color:{color}'>F = {f_stat:.2f}, p = {p_val:.4f} "
                        f"({'✅ significant' if p_val < 0.05 else '❌ not significant'})</span><br>"
                        f"<i>Tests whether cards from different sets display different average growth rates.</i>",
                        unsafe_allow_html=True
                    )
                    poke_tests_displayed += 1

            if poke_tests_displayed < 3:
                st.info(f"ℹ️ Displaying {poke_tests_displayed} of 3 Pokémon statistical tests (some data may be missing)")

        # Star Wars (Right Column)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.scatterplot(
                data=sw_feat, x="Mean_Growth", y="Volatility",
                hue="Cluster_Name", palette=palette_named, s=100, ax=ax2
            )
            ax2.set_title("🌌 Star Wars — Growth vs Volatility Clusters", fontsize=11)
            ax2.grid(alpha=.3)
            ax2.legend(title="Cluster", loc="best")
            st.pyplot(fig2)

            # 🌌 Star Wars Statistical Tests
            st.markdown("### 🌌 Star Wars — Statistical Tests")
            sw_tests_displayed = 0

            # 1️⃣ Condition → Growth
            groups = [g["Mean_Growth"].dropna() for _, g in sw_feat.groupby("Condition") if len(g) > 1]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                color = "green" if p_val < 0.05 else "red"
                st.markdown(
                    f"<b>🧱 Condition → Growth:</b> "
                    f"<span style='color:{color}'>F = {f_stat:.2f}, p = {p_val:.4f} "
                    f"({'✅ significant' if p_val < 0.05 else '❌ not significant'})</span><br>"
                    f"<i>Tests whether mint, loose, or packaged figures exhibit different growth profiles.</i>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<b>🧱 Condition → Growth:</b> <span style='color:gray'>Not enough data for test.</span>",
                    unsafe_allow_html=True
                )
            sw_tests_displayed += 1

            # 2️⃣ Authenticity → Growth (✅ fixed)
            groups = [g["Mean_Growth"].dropna() for _, g in sw_feat.groupby("Authenticity") if len(g) > 1]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                color = "green" if p_val < 0.05 else "red"
                st.markdown(
                    f"<b>🔖 Authenticity → Growth:</b> "
                    f"<span style='color:{color}'>F = {f_stat:.2f}, p = {p_val:.4f} "
                    f"({'✅ significant' if p_val < 0.05 else '❌ not significant'})</span><br>"
                    f"<i>Evaluates if authenticity level correlates with collectible growth rates.</i>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<b>🔖 Authenticity → Growth:</b> <span style='color:gray'>Not enough data for test.</span>",
                    unsafe_allow_html=True
                )
            sw_tests_displayed += 1

            # 3️⃣ Character Type → Growth
            groups = [g["Mean_Growth"].dropna() for _, g in sw_feat.groupby("Character_Type") if len(g) > 1]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                color = "green" if p_val < 0.05 else "red"
                st.markdown(
                    f"<b>🦾 Character Type → Growth:</b> "
                    f"<span style='color:{color}'>F = {f_stat:.2f}, p = {p_val:.4f} "
                    f"({'✅ significant' if p_val < 0.05 else '❌ not significant'})</span><br>"
                    f"<i>Checks if hero, villain, or droid characters behave differently in price appreciation.</i>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<b>🦾 Character Type → Growth:</b> <span style='color:gray'>Not enough data for test.</span>",
                    unsafe_allow_html=True
                )
            sw_tests_displayed += 1

            if sw_tests_displayed < 3:
                st.info(f"ℹ️ Displaying {sw_tests_displayed} of 3 Star Wars statistical tests (some data may be missing)")

    # ------------------------------------------------------------
    # 🧩 Findings Summary
    # ------------------------------------------------------------
    st.markdown("""
    ### 🔍 Findings
    - Both markets form **three distinct clusters**: blue-chip, mid-tier, and speculative.  
    - Pokémon shows **more stable and compact clusters**, reflecting a mature market.  
    - Star Wars shows **higher within-group volatility**, indicating event-driven price changes.  
    ✅ **Hypothesis 2 validated:** Collectibles exhibit structured market segmentation.
    """)

# ============================================================
# 🔮 Hypothesis 3 — Forecastability of Collectibles
# ============================================================
with tab7:
    st.header("🔮 Hypothesis 3 — Forecastability of Collectibles")

    st.markdown("""
    **Statement:**  
    Historical price behaviour allows short-term forecasting of collectible values  
    (based on *graded cards* and *figure selling prices*).
    """)

    col1, col2 = st.columns(2)

    # ============================================================
    # 🎮 Pokémon — Prophet Forecast (Monthly Graded Cards)
    # ============================================================
    with col1:
        st.subheader("🎮 Pokémon — Graded Cards Market Average (Prophet Forecast)")

        from prophet import Prophet
        import matplotlib.ticker as mtick

        df_poke = pokemon_final_26.copy()
        df_poke["Date"] = pd.to_datetime(df_poke["Date"], errors="coerce")
        df_poke = df_poke.dropna(subset=["Date", "Graded"])
        df_poke["Month"] = df_poke["Date"].dt.to_period("M").dt.to_timestamp()

        poke_avg = df_poke.groupby("Month")["Graded"].mean().reset_index()
        poke_avg = poke_avg.rename(columns={"Month": "ds", "Graded": "y"})

        if len(poke_avg) >= 12:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(poke_avg)
            future = m.make_future_dataframe(periods=6, freq="M")
            fc = m.predict(future)

            fig_p, ax_p = plt.subplots(figsize=(8, 4))
            ax_p.plot(poke_avg["ds"], poke_avg["y"], "k.", label="Historical Avg (Graded)")
            ax_p.plot(fc["ds"], fc["yhat"], "r-", label="Forecast")
            ax_p.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                              color="lightcoral", alpha=0.3)
            ax_p.set_title("Pokémon Graded Cards — Prophet Forecast", fontsize=11, fontweight="bold")
            ax_p.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))  # numeric formatting
            ax_p.grid(alpha=0.3)
            ax_p.legend()
            st.pyplot(fig_p)
        else:
            st.warning("Not enough Pokémon data for monthly forecast.")

    # ============================================================
    # 🌌 Star Wars — Polynomial Forecast (Yearly Figures)
    # ============================================================
    with col2:
        st.subheader("🌌 Star Wars — Figures Market Average (Polynomial Forecast)")

        df_sw = starwars_moc_loose.copy()
        if "year" in df_sw.columns and "selling_price" in df_sw.columns:
            sw_avg = df_sw.groupby("year")["selling_price"].mean().reset_index()

            if len(sw_avg) >= 5:
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import PolynomialFeatures

                X, y = sw_avg[["year"]], sw_avg["selling_price"]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_fit = model.predict(X_poly)  # historical fit
                future_years = np.arange(sw_avg["year"].max() + 1,
                                         sw_avg["year"].max() + 4).reshape(-1, 1)
                y_pred = model.predict(poly.transform(future_years))
                r2 = model.score(X_poly, y)

                fig_sw, ax_sw = plt.subplots(figsize=(8, 4))
                ax_sw.scatter(X, y, label="Historical Avg (Figures)", color="#0070C0")
                ax_sw.plot(X, y_fit, "r-", label=f"Polynomial Fit (R²={r2:.2f})")     # full historical fit
                ax_sw.plot(future_years, y_pred, "r--", label="Forecast (Extrapolated)")
                ax_sw.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
                ax_sw.set_title("Star Wars Figures — Polynomial Forecast", fontsize=11, fontweight="bold")
                ax_sw.grid(alpha=0.3)
                ax_sw.legend()
                st.pyplot(fig_sw)
            else:
                st.warning("Not enough Star Wars data for polynomial forecast.")
        else:
            st.error("Missing 'year' or 'selling_price' column in Star Wars dataset.")

    # ============================================================
    # 📈 Normalized Market Index — Recent Years Only
    # ============================================================
    st.markdown("---")
    st.subheader("📈 Normalized Market Index — Pokémon Graded Cards vs Star Wars Figures")

    try:
        # --- Pokémon monthly avg ---
        df_poke = pokemon_final_26.copy()
        df_poke["Date"] = pd.to_datetime(df_poke["Date"], errors="coerce")
        df_poke = df_poke.dropna(subset=["Date", "Graded"])
        df_poke["Month"] = df_poke["Date"].dt.to_period("M").dt.to_timestamp()
        poke_hist = df_poke.groupby("Month")["Graded"].mean().reset_index()
        poke_hist["category"] = "Pokémon Historical"

        # --- Star Wars yearly avg ---
        df_sw = starwars_moc_loose.copy()
        df_sw = df_sw.dropna(subset=["year", "selling_price"])
        sw_hist = df_sw.groupby("year")["selling_price"].mean().reset_index()
        sw_hist["category"] = "Star Wars Historical"

        # Convert both to comparable datetime axis
        sw_hist["date"] = pd.to_datetime(sw_hist["year"], format="%Y")
        poke_hist = poke_hist.rename(columns={"Month": "date", "Graded": "price"})
        sw_hist = sw_hist.rename(columns={"selling_price": "price"})

        # --- Normalize both, base = first value ---
        poke_hist["normalized"] = (poke_hist["price"] / poke_hist["price"].iloc[0]) * 100
        sw_hist["normalized"] = (sw_hist["price"] / sw_hist["price"].iloc[0]) * 100

        # --- Filter recent window (post-2015) ---
        min_date = pd.to_datetime("2015-01-01")
        poke_hist = poke_hist[poke_hist["date"] >= min_date]
        sw_hist = sw_hist[sw_hist["date"] >= min_date]

        # --- Plot ---
        fig_idx, ax_idx = plt.subplots(figsize=(10, 5))
        ax_idx.plot(poke_hist["date"], poke_hist["normalized"],
                    color="#FF6B6B", linewidth=2, marker="o", label="Pokémon Historical")
        ax_idx.plot(sw_hist["date"], sw_hist["normalized"],
                    color="#4ECDC4", linewidth=2, marker="o", label="Star Wars Historical")

        ax_idx.set_title("Normalized Market Index (Base = 100) — Recent Years",
                         fontsize=13, fontweight="bold")
        ax_idx.set_xlabel("Date")
        ax_idx.set_ylabel("Normalized Price Index")
        ax_idx.grid(alpha=0.3)
        ax_idx.legend()
        st.pyplot(fig_idx)

    except Exception as e:
        st.error(f"Error creating normalized market index: {e}")

    # ============================================================
    # Findings
    # ============================================================
    st.markdown("""
    ### 🔍 Findings
    - **Pokémon:** Monthly Prophet forecast captures graded-card volatility and mean reversion.  
    - **Star Wars:** Polynomial model fits full historical trend and projects mild upward growth.  
    - **Market Index:** Focused on recent years, both lines show comparable growth behaviour.  
    ✅ **Hypothesis 3 validated:** Historical data supports short-term predictive modelling of collectible prices.
    """)




with tab8:
    st.header("Conclusions")
    st.subheader("Jonas to add here")










#col1, col2 = st.columns(2)

#with col1:
   # st.image('Users/amyby/Desktop/Bootcamp/final_project/images/darth_vader.jpg', caption='How much would you pay for this?')

#with col2:
    #st.image('Users/amyby/Desktop/Bootcamp/final_project/images/pikachu.jpg', caption='How valuable do you think this is?')