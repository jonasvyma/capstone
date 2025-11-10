"""
Pokemon Card Pricing Data Analysis
A comprehensive exploratory data analysis of Pokemon card pricing data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
df = pd.read_csv('../data/pokemon/final_dataset.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.to_period('M')

print("=" * 80)
print("POKEMON CARD PRICING DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. DATA OVERVIEW
# ============================================================================
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total records: {len(df):,}")
print(f"Date range: {df['Date'].min().strftime('%B %Y')} - {df['Date'].max().strftime('%B %Y')}")
print(f"Total months: {df['Date'].nunique()}")
print(f"\nColumns: {', '.join(df.columns.tolist())}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n2. BASIC STATISTICS")
print("-" * 80)

# Price columns
price_cols = ['Boxonly', 'Cib', 'Graded', 'Manualonly', 'New', 'Used']
print("\nPrice columns statistics:")
print(df[price_cols].describe())

print("\n\nCategorical Variables:")
print(f"  - Unique cards: {df['Card Name'].nunique()}")
print(f"  - Unique rarities: {df['Rarity'].nunique()}")
print(f"  - Unique sets: {df['Set Name'].nunique()}")
print(f"  - Unique illustrators: {df['Illustrator'].nunique()}")

print(f"\n  - Rarity distribution:")
for rarity, count in df['Rarity'].value_counts().head(10).items():
    print(f"    • {rarity}: {count}")

# ============================================================================
# 3. VISUALIZATIONS
# ============================================================================

# Create a figure directory if it doesn't exist
if not os.path.exists('pk_plots'):
    os.makedirs('pk_plots')

# Plot 1: Price Distribution Across All Conditions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Price Distribution by Condition', fontsize=16, fontweight='bold')

for idx, col in enumerate(price_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]

    # Filter out zeros for better visualization
    data = df[df[col] > 0][col]

    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{col} (n={len(data):,})')
    ax.axvline(data.mean(), color='red', linestyle='--',
               label=f'Mean: ${data.mean():,.0f}')
    ax.axvline(data.median(), color='green', linestyle='--',
               label=f'Median: ${data.median():,.0f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('pk_plots/01_price_distributions.png', dpi=300, bbox_inches='tight')
print("\n3. VISUALIZATIONS")
print("-" * 80)
print("✓ Saved: 01_price_distributions.png")
plt.close()

# Plot 2: Average Price by Condition (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 6))
avg_prices = []
median_prices = []
for col in price_cols:
    non_zero = df[df[col] > 0][col]
    avg_prices.append(non_zero.mean() if len(non_zero) > 0 else 0)
    median_prices.append(non_zero.median() if len(non_zero) > 0 else 0)

x = np.arange(len(price_cols))
width = 0.35

ax.bar(x - width/2, avg_prices, width, label='Mean Price', alpha=0.8)
ax.bar(x + width/2, median_prices, width, label='Median Price', alpha=0.8)
ax.set_xlabel('Condition')
ax.set_ylabel('Price ($)')
ax.set_title('Average Price by Card Condition (Excluding Zeros)')
ax.set_xticks(x)
ax.set_xticklabels(price_cols, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Format y-axis as currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('pk_plots/02_avg_price_by_condition.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_avg_price_by_condition.png")
plt.close()

# Plot 3: Price by Rarity (Top 8 Rarities)
fig, ax = plt.subplots(figsize=(14, 7))
top_rarities = df['Rarity'].value_counts().head(8).index

# Calculate average price across all conditions for each rarity
rarity_prices = []
for rarity in top_rarities:
    rarity_df = df[df['Rarity'] == rarity]
    # Average across all non-zero price columns
    avg_price = rarity_df[price_cols].replace(0, np.nan).mean().mean()
    rarity_prices.append(avg_price)

colors = sns.color_palette('husl', len(top_rarities))
bars = ax.bar(range(len(top_rarities)), rarity_prices, color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('Rarity')
ax.set_ylabel('Average Price ($)')
ax.set_title('Average Price by Rarity (Top 8 Rarity Types)')
ax.set_xticks(range(len(top_rarities)))
ax.set_xticklabels(top_rarities, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('pk_plots/03_price_by_rarity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_price_by_rarity.png")
plt.close()

# Plot 4: Price Trends Over Time (Multi-line for each condition)
fig, ax = plt.subplots(figsize=(16, 7))

for col in price_cols:
    monthly_avg = df.groupby('Date')[col].mean()
    ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2,
            label=col, markersize=3, alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('Average Price ($)')
ax.set_title('Pokemon Card Price Trends Over Time by Condition')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pk_plots/04_price_trends_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_price_trends_over_time.png")
plt.close()

# Plot 5: Top 15 Most Valuable Cards (by average of all conditions)
fig, ax = plt.subplots(figsize=(12, 8))

card_avg_prices = df.groupby('Card Name')[price_cols].mean().mean(axis=1).sort_values(ascending=False).head(15)

ax.barh(range(len(card_avg_prices)), card_avg_prices.values, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(card_avg_prices)))
ax.set_yticklabels(card_avg_prices.index)
ax.set_xlabel('Average Price Across All Conditions ($)')
ax.set_title('Top 15 Most Valuable Pokemon Cards')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig('pk_plots/05_top_15_cards.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_top_15_cards.png")
plt.close()

# Plot 6: Top 10 Sets by Average Price
fig, ax = plt.subplots(figsize=(12, 7))

set_avg_prices = df.groupby('Set Name')[price_cols].mean().mean(axis=1).sort_values(ascending=False).head(10)

colors = sns.color_palette('viridis', len(set_avg_prices))
bars = ax.bar(range(len(set_avg_prices)), set_avg_prices.values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('Set Name')
ax.set_ylabel('Average Price ($)')
ax.set_title('Top 10 Pokemon Sets by Average Price')
ax.set_xticks(range(len(set_avg_prices)))
ax.set_xticklabels(set_avg_prices.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('pk_plots/06_top_10_sets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_top_10_sets.png")
plt.close()

# Plot 7: Condition Comparison Boxplot
fig, ax = plt.subplots(figsize=(14, 7))

# Prepare data for boxplot (melt the dataframe)
price_data = []
conditions = []
for col in price_cols:
    non_zero = df[df[col] > 0][col]
    price_data.extend(non_zero.values)
    conditions.extend([col] * len(non_zero))

boxplot_df = pd.DataFrame({'Price': price_data, 'Condition': conditions})

sns.boxplot(data=boxplot_df, x='Condition', y='Price', ax=ax)
ax.set_xlabel('Condition')
ax.set_ylabel('Price ($)')
ax.set_title('Price Distribution by Condition (Excluding Zeros)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yscale('log')  # Log scale due to high variance
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig('pk_plots/07_condition_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_condition_boxplot.png")
plt.close()

# Plot 8: Correlation Heatmap (Price Conditions)
fig, ax = plt.subplots(figsize=(10, 8))
correlation = df[price_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of Price Conditions')
plt.tight_layout()
plt.savefig('pk_plots/08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_correlation_heatmap.png")
plt.close()

# Plot 9: Rarity Distribution (Pie Chart)
fig, ax = plt.subplots(figsize=(10, 8))
rarity_counts = df['Rarity'].value_counts().head(10)
colors = sns.color_palette('Set3', len(rarity_counts))
wedges, texts, autotexts = ax.pie(rarity_counts.values, labels=rarity_counts.index,
                                    autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title('Distribution of Card Rarities (Top 10)')
plt.setp(autotexts, size=8, weight="bold")
plt.setp(texts, size=9)
plt.tight_layout()
plt.savefig('pk_plots/09_rarity_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_rarity_distribution.png")
plt.close()

# Plot 10: Monthly Record Count Over Time
fig, ax = plt.subplots(figsize=(14, 6))
monthly_counts = df.groupby('Date').size()
ax.bar(monthly_counts.index, monthly_counts.values, alpha=0.7, edgecolor='black', width=20)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Records')
ax.set_title('Data Volume Over Time')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pk_plots/10_data_volume_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 10_data_volume_over_time.png")
plt.close()

# ============================================================================
# 4. KEY INSIGHTS
# ============================================================================
print("\n4. KEY INSIGHTS")
print("-" * 80)

# Most valuable card
card_avg = df.groupby('Card Name')[price_cols].mean().mean(axis=1)
most_valuable = card_avg.idxmax()
print(f"Most valuable card (avg across conditions): {most_valuable} (${card_avg[most_valuable]:,.2f})")

# Highest single price
max_prices = {}
for col in price_cols:
    max_val = df[df[col] > 0][col].max()
    max_prices[col] = max_val

highest_condition = max(max_prices, key=max_prices.get)
print(f"\nHighest recorded price: ${max_prices[highest_condition]:,.2f} ({highest_condition})")

# Average prices by condition
print(f"\nAverage price by condition (excluding zeros):")
for col in price_cols:
    non_zero = df[df[col] > 0][col]
    if len(non_zero) > 0:
        print(f"  - {col}: ${non_zero.mean():,.2f} (median: ${non_zero.median():,.2f})")

# Top 5 most valuable rarities
print(f"\nTop 5 most valuable rarities:")
rarity_avg = df.groupby('Rarity')[price_cols].mean().mean(axis=1).sort_values(ascending=False)
for i, (rarity, price) in enumerate(rarity_avg.head(5).items(), 1):
    print(f"  {i}. {rarity}: ${price:,.2f}")

# Top 5 most valuable sets
print(f"\nTop 5 most valuable sets:")
set_avg = df.groupby('Set Name')[price_cols].mean().mean(axis=1).sort_values(ascending=False)
for i, (set_name, price) in enumerate(set_avg.head(5).items(), 1):
    print(f"  {i}. {set_name}: ${price:,.2f}")

# Most frequent cards
print(f"\nTop 5 most tracked cards (by number of records):")
for i, (card, count) in enumerate(df['Card Name'].value_counts().head(5).items(), 1):
    avg_price = df[df['Card Name'] == card][price_cols].mean().mean()
    print(f"  {i}. {card}: {count} records (avg: ${avg_price:,.2f})")

# Price trends
print(f"\nPrice trends:")
for col in price_cols:
    non_zero = df[df[col] > 0]
    if len(non_zero) > 10:
        corr = non_zero[col].corr(non_zero['Date'].astype(np.int64))
        trend = "increasing" if corr > 0.1 else ("decreasing" if corr < -0.1 else "stable")
        print(f"  - {col}: {trend} (correlation: {corr:.3f})")

# ============================================================================
# 5. STATISTICAL ANALYSIS
# ============================================================================
print("\n5. STATISTICAL ANALYSIS")
print("-" * 80)

# ANOVA test across conditions
print("ANOVA test across all conditions (excluding zeros):")
condition_data = [df[df[col] > 0][col].values for col in price_cols]
condition_data = [data for data in condition_data if len(data) > 0]
f_stat, p_value = stats.f_oneway(*condition_data)
print(f"  F-statistic={f_stat:.4f}, p-value={p_value:.4e}")
print(f"  → {'Significant' if p_value < 0.05 else 'Not significant'} differences between conditions")

# Pairwise comparisons (sample: New vs Used)
if len(df[df['New'] > 0]) > 0 and len(df[df['Used'] > 0]) > 0:
    new_prices = df[df['New'] > 0]['New']
    used_prices = df[df['Used'] > 0]['Used']
    t_stat, p_value = stats.ttest_ind(new_prices, used_prices)
    print(f"\nT-test (New vs Used): t-statistic={t_stat:.4f}, p-value={p_value:.4e}")
    print(f"  → {'Significant' if p_value < 0.05 else 'Not significant'} difference")

# Price correlation between conditions
print(f"\nHighest price correlations between conditions:")
corr_matrix = df[price_cols].corr()
# Get upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
corr_pairs = []
for i in range(len(price_cols)):
    for j in range(i+1, len(price_cols)):
        corr_pairs.append((price_cols[i], price_cols[j], corr_matrix.iloc[i, j]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for pair in corr_pairs[:5]:
    print(f"  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")

print("\n" + "=" * 80)
print("Analysis complete! All plots saved to 'pk_plots/' directory")
print("=" * 80)
