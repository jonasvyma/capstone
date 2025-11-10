import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Collectibles Analysis", layout="wide")

# Set consistent style
sns.set_style('whitegrid')
sns.set_palette(['steelblue'])
plt.rcParams['figure.figsize'] = (14, 8)

# Title
st.title("Collectibles Pricing Analysis")

# Create tabs
tab1, tab2 = st.tabs(["Pokemon Cards", "Star Wars Figures"])

# ========================================
# TAB 1: POKEMON CARDS
# ========================================
with tab1:
    st.header("Pokemon Card Pricing Analysis")

    # Load the data
    df = pd.read_csv('final_dataset.csv')

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['YearMonth'] = df['Date'].dt.to_period('M')

    # Filter out Booster Box outliers
    df = df[df['Rarity'] != 'Booster Box']

    # Filter out Booster Pack/Box products from card names
    df = df[~df['Card Name'].str.contains('Booster|Pack|Box', case=False, na=False)]

    # Define price columns
    price_cols = ['New', 'Used', 'Graded']

    # Data Overview
    st.subheader("Data Overview")
    st.write(f"📊 Pokemon Cards: {len(df):,} records | {df['Card Name'].nunique()} cards | {df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}")

    st.markdown("---")

    # Top 15 Most Valuable Pokemon Cards
    st.subheader("Top 15 Most Valuable Pokemon Cards")
    card_avg_prices = df[df['Graded'] > 0].groupby('Card Name')['Graded'].mean().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(card_avg_prices)), card_avg_prices.values,
            alpha=0.8, edgecolor='black', color='steelblue')
    ax.set_yticks(range(len(card_avg_prices)))
    ax.set_yticklabels(card_avg_prices.index)
    ax.set_xlabel('Average Graded Price ($)')
    ax.set_title('Top 15 Most Valuable Pokemon Cards (Graded)')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Top 10 Pokemon Sets by Average Graded Price
    st.subheader("Top 10 Pokemon Sets by Average Graded Price")
    set_avg_prices = df[df['Graded'] > 0].groupby('Set Name')['Graded'].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(set_avg_prices)), set_avg_prices.values,
                   color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Set Name')
    ax.set_xlabel('Average Graded Price ($)')
    ax.set_title('Top 10 Pokemon Sets by Average Graded Price')
    ax.set_yticks(range(len(set_avg_prices)))
    ax.set_yticklabels(set_avg_prices.index)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Price Distribution by Condition (Boxplot)
    st.subheader("Price Distribution by Condition")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data for boxplot
    price_data = []
    conditions = []
    for col in price_cols:
        non_zero = df[df[col] > 0][col]
        price_data.extend(non_zero.values)
        conditions.extend([col] * len(non_zero))

    boxplot_df = pd.DataFrame({'Price': price_data, 'Condition': conditions})

    sns.boxplot(data=boxplot_df, x='Condition', y='Price', ax=ax, color='steelblue')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price Distribution by Condition')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Pokemon Card Growth Analysis")
    st.markdown("---")

    # Load Pokemon data for growth analysis
    df_pk = pd.read_csv('final_dataset.csv')
    df_pk['Date'] = pd.to_datetime(df_pk['Date'])

    # Filter for graded prices and exclude booster products
    df_pk = df_pk[df_pk['Graded'] > 0]
    df_pk = df_pk[~df_pk['Card Name'].str.contains('Booster|Pack|Box', case=False, na=False)]

    # Calculate growth for each card
    growth_results = []
    for card in df_pk['Card Name'].unique():
        card_data = df_pk[df_pk['Card Name'] == card].sort_values('Date')
        if len(card_data) >= 2:
            first_price = card_data.iloc[0]['Graded']
            last_price = card_data.iloc[-1]['Graded']
            first_date = card_data.iloc[0]['Date']
            last_date = card_data.iloc[-1]['Date']
            if first_price > 0:
                pct_growth = ((last_price - first_price) / first_price) * 100
                growth_results.append({
                    'Card': card,
                    'First Price': first_price,
                    'Last Price': last_price,
                    'Growth %': pct_growth,
                    'Months': (last_date - first_date).days / 30.44
                })

    pk_growth_df = pd.DataFrame(growth_results).sort_values('Growth %', ascending=False)
    st.write(f"**Top 5 Cards by Growth:**")
    st.dataframe(pk_growth_df.head())

    # Visualize top 15 growers
    fig, ax = plt.subplots(figsize=(12, 8))

    top_15 = pk_growth_df.head(15).sort_values('Growth %')
    colors = ['green' if x > 0 else 'red' for x in top_15['Growth %']]

    ax.barh(range(len(top_15)), top_15['Growth %'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['Card'])
    ax.set_xlabel('Growth %', fontsize=12)
    ax.set_title('Top 15 Pokemon Cards by Price Growth % (Graded)', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, v in enumerate(top_15['Growth %']):
        ax.text(v + 5 if v > 0 else v - 5, i, f"{v:.1f}%",
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Price Evolution Over Time - Top 3 Pokemon Cards
    st.subheader("Price Evolution: Top 3 Pokemon Cards by Growth")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 3 cards by growth
    top_3_cards = pk_growth_df.head(3)['Card'].values

    # Define distinct colors for each line
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for idx, card in enumerate(top_3_cards):
        card_data = df_pk[df_pk['Card Name'] == card].sort_values('Date')
        ax.plot(card_data['Date'], card_data['Graded'],
                marker='o', linewidth=2.5, markersize=6,
                label=card, color=line_colors[idx], alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Graded Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Price Evolution: Top 3 Pokemon Cards by Growth',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========================================
# TAB 2: STAR WARS FIGURES
# ========================================
with tab2:
    st.header("Star Wars Action Figures Analysis")

    # Load the data
    df = pd.read_csv('starwars_filtered.csv')

    # Data Overview
    graded_pct = (df['authenticity_n'] == 1).sum() / len(df) * 100
    st.subheader("Data Overview")
    st.write(f"⚔️ Star Wars Figures: {len(df):,} records | {df['figure'].nunique()} figures | {df['year'].min()}-{df['year'].max()} | {graded_pct:.0f}% graded")

    st.markdown("---")

    # Top 10 Most Valuable Figures by Grading Status
    st.subheader("Top 10 Most Valuable Figures by Grading Status")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    combinations = [
        (1, 'Graded/Certified', axes[0]),
        (0, 'Not Graded', axes[1])
    ]

    for auth, title, ax in combinations:
        subset = df[df['authenticity_n'] == auth]

        figure_stats = subset.groupby('figure').agg({
            'selling_price': ['mean', 'count']
        })
        figure_stats.columns = ['avg_price', 'count']
        figure_stats = figure_stats[figure_stats['count'] >= 5]
        top_10 = figure_stats.nlargest(10, 'avg_price')

        y_pos = np.arange(len(top_10))
        ax.barh(y_pos, top_10['avg_price'], color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_10.index, fontsize=9)
        ax.set_xlabel('Average Selling Price ($)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        for i, (idx, row) in enumerate(top_10.iterrows()):
            ax.text(row['avg_price'] + 10, i, f"n={int(row['count'])}",
                    va='center', fontsize=10, color='black')

    fig.suptitle('Top 10 Most Valuable Figures by Grading Status (min 5 sales)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Top 10 Best Selling Figures by Grading Status
    st.subheader("Top 10 Best Selling Figures by Grading Status")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    combinations = [
        (1, 'Graded/Certified', axes[0]),
        (0, 'Not Graded', axes[1])
    ]

    for auth, title, ax in combinations:
        subset = df[df['authenticity_n'] == auth]

        figure_sales = subset.groupby('figure')['sales'].sum().sort_values(ascending=False).head(10)

        avg_prices = []
        for figure in figure_sales.index:
            avg_price = subset[subset['figure'] == figure]['selling_price'].mean()
            avg_prices.append(avg_price)

        plot_data = pd.DataFrame({
            'figure': figure_sales.index,
            'total_sales': figure_sales.values,
            'avg_price': avg_prices
        })

        y_pos = np.arange(len(plot_data))
        ax.barh(y_pos, plot_data['total_sales'], color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data['figure'], fontsize=9)
        ax.set_xlabel('Total Sales Volume', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        for i, (idx, row) in enumerate(plot_data.iterrows()):
            ax.text(row['total_sales'] + max(plot_data['total_sales'])*0.02, i,
                    f"${row['avg_price']:.0f}",
                    va='center', fontsize=10, color='black', fontweight='normal')

    fig.suptitle('Top 10 Best Selling Figures by Grading Status',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Price Distribution by Grading (Boxplot)
    st.subheader("Price Distribution by Grading Status")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data for boxplot
    price_data = []
    grading_status = []

    for auth in [1, 0]:
        data = df[df['authenticity_n'] == auth]['selling_price']
        data = data[data > 0]
        price_data.extend(data.values)
        label = 'Graded/Certified' if auth == 1 else 'Not Graded'
        grading_status.extend([label] * len(data))

    boxplot_df = pd.DataFrame({'Price': price_data, 'Grading Status': grading_status})

    sns.boxplot(data=boxplot_df, x='Grading Status', y='Price', ax=ax, color='steelblue')
    ax.set_xlabel('Grading Status')
    ax.set_ylabel('Selling Price ($)')
    ax.set_title('Star Wars Price Distribution by Grading Status')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Star Wars Figure Growth Analysis")
    st.markdown("---")

    # Load Star Wars data for growth analysis
    df_sw = pd.read_csv('starwars_filtered.csv')

    # Filter for authentic (graded) figures only
    df_sw = df_sw[df_sw['authenticity_n'] == 1]
    df_sw = df_sw[df_sw['selling_price'] > 0]

    # Calculate growth for each figure
    sw_growth_results = []
    for figure in df_sw['figure'].unique():
        figure_data = df_sw[df_sw['figure'] == figure].sort_values('year')
        if len(figure_data) >= 2:
            first_year = figure_data['year'].min()
            last_year = figure_data['year'].max()
            first_year_data = figure_data[figure_data['year'] == first_year]
            last_year_data = figure_data[figure_data['year'] == last_year]
            first_price = first_year_data['selling_price'].mean()
            last_price = last_year_data['selling_price'].mean()
            if first_price > 0:
                pct_growth = ((last_price - first_price) / first_price) * 100
                sw_growth_results.append({
                    'Figure': figure,
                    'First Price': first_price,
                    'Last Price': last_price,
                    'Growth %': pct_growth,
                    'Years': last_year - first_year
                })

    sw_growth_df = pd.DataFrame(sw_growth_results).sort_values('Growth %', ascending=False)
    st.write(f"**Top 5 Figures by Growth:**")
    st.dataframe(sw_growth_df.head())

    # Visualize top 15 growers
    fig, ax = plt.subplots(figsize=(12, 8))

    top_15_sw = sw_growth_df.head(15).sort_values('Growth %')
    colors = ['green' if x > 0 else 'red' for x in top_15_sw['Growth %']]

    ax.barh(range(len(top_15_sw)), top_15_sw['Growth %'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_15_sw)))
    ax.set_yticklabels(top_15_sw['Figure'], fontsize=9)
    ax.set_xlabel('Growth %', fontsize=12)
    ax.set_title('Top 15 Star Wars Figures by Price Growth % (Authentic)', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, v in enumerate(top_15_sw['Growth %']):
        ax.text(v + max(top_15_sw['Growth %'])*0.02 if v > 0 else v - max(top_15_sw['Growth %'])*0.02,
                i, f"{v:.1f}%", va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Price Evolution Over Time - Top 3 Star Wars Figures
    st.subheader("Price Evolution: Top 3 Star Wars Figures by Growth (Authentic)")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 3 figures by growth
    top_3_figures = sw_growth_df.head(3)['Figure'].values

    # Define distinct colors for each line (same as Pokemon)
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for idx, figure in enumerate(top_3_figures):
        figure_data = df_sw[df_sw['figure'] == figure].groupby('year')['selling_price'].mean().reset_index()
        ax.plot(figure_data['year'], figure_data['selling_price'],
                marker='o', linewidth=2.5, markersize=6,
                label=figure, color=line_colors[idx], alpha=0.8)

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Selling Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Price Evolution: Top 3 Star Wars Figures by Growth (Authentic)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
