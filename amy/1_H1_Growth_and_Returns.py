import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker

# Title
st.title('Are these collectibles a long term investment?')
st.header('Testing whether cards and figures appreciate over time and deliver an average return over 2%.')

# Reading in data and defining variables
starwars_moc_loose = pd.read_csv('C:/Users/amyby/Desktop/Bootcamp/final_project/data/starwars_moc_loose_v7.csv')
pokemon_final_26 = pd.read_csv('C:/Users/amyby/Desktop/Bootcamp/final_project/data/final_dataset_csv.csv')
pokemon_medians = pd.read_csv('C:/Users/amyby/Desktop/Bootcamp/final_project/pokemons_medians_df.csv')
starwars_medians = pd.read_csv('C:/Users/amyby/Desktop/Bootcamp/final_project/starwars_medians_df.csv')
starwars_year_figure = pd.read_csv('C:/Users/amyby/Desktop/Bootcamp/final_project/starwars_year_figure.csv')
pokemon_year_card = pd.read_csv('C:/Users/amyby/Desktop/Bootcamp/final_project/pokemon_year_card_df.csv')
pokemon_returns = pokemon_year_card['YoY_Growth'].replace([np.inf, -np.inf], np.nan).dropna()
starwars_returns = starwars_year_figure['YoY_Growth'].dropna()
sw_returns_list = starwars_returns.to_list()
poke_returns_list = pokemon_returns.to_list()


# T-tests
exp_return = 0.02
sw_t_stat, sw_p_value = stats.ttest_1samp(sw_returns_list, exp_return)
poke_t_stat, poke_p_value = stats.ttest_1samp(poke_returns_list, exp_return)

test_results_df = pd.DataFrame({
    'Franchise': ['Star Wars', 'Pokemon'],
    'T-Statistic': [sw_t_stat, poke_t_stat],
    'P-Value': [sw_p_value, poke_p_value]
})

st.error("**H₀:** Star Wars and Pokémon, on average, do not exceed the expected annual return of 2%. ❌ Rejected")
st.success("**H₁:** Star Wars and Pokémon, on average, achieve an annual return greater than 2%. ✅ Accepted")

with st.expander("T-Test Results for H1"):
    st.dataframe(test_results_df)

# Plot Star Wars
st.subheader("Star Wars Figures Median Selling Price Development")
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=starwars_medians, x='year', y='selling_price', ax=ax1, marker='o', color='blue', label='Median Selling Price')
sns.lineplot(data=starwars_medians, x='year', y='expected_price', ax=ax1, marker='o', color='red', label='Expected Price\n with 2% return p.a')

ax1.set_ylabel('Selling Price (USD)', color='blue')
ax1.set_xlabel('Year')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

ax2 = ax1.twinx()
sns.lineplot(data=starwars_medians, x='year', y='YoY_Growth', ax=ax2, marker='o', color='green', label='YoY Growth', alpha=0.3)
ax2.set_ylabel('YoY Growth (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y * 10:.1f}%'))

ax1.grid(True)
ax1.legend(loc='upper left')
st.pyplot(fig)

# Plot Pokemon 
# Plot Pokemon
st.subheader("Pokemon Graded Price Development")
fig2, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=pokemon_medians, x='year', y='Graded', ax=ax1, marker='o', color='blue', label='Median Selling Price')
sns.lineplot(data=pokemon_medians, x='year', y='expected_price', ax=ax1, marker='o', color='red', label='Expected Price\nwith 2% return p.a')

ax1.set_ylabel('Selling Price (USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

# Second axis
ax2 = ax1.twinx()
sns.lineplot(data=pokemon_medians, x='year', y='YoY_Growth', ax=ax2, marker='o', color='green', label='YoY Growth', alpha=0.5)
ax2.set_ylabel('YoY Growth (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y * 10:.1f}%'))
ax2.legend(loc='upper right')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

ax1.grid(True)
st.pyplot(fig2)
