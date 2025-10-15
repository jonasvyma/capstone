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
    st.header("Hypothesis 2 - Findings")
    st.subheader("Amy to add here")

with tab7:
    st.header("Hypothesis 3 - Findings")
    st.subheader("Simo to add here")

with tab8:
    st.header("Conclusions")
    st.subheader("Jonas to add here")

#col1, col2 = st.columns(2)

#with col1:
   # st.image('Users/amyby/Desktop/Bootcamp/final_project/images/darth_vader.jpg', caption='How much would you pay for this?')

#with col2:
    #st.image('Users/amyby/Desktop/Bootcamp/final_project/images/pikachu.jpg', caption='How valuable do you think this is?')