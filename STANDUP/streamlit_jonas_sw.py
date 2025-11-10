import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Page configuration
st.set_page_config(page_title="Star Wars Action Figures Analysis", layout="wide")

# Set style for plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Title
st.title("Star Wars Action Figures Data Analysis")
st.markdown("## Comprehensive Exploratory Data Analysis")
st.markdown("This dashboard analyzes Star Wars action figure sales data including pricing trends, character types, and market dynamics.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('starwars_filtered.csv')
    return df

try:
    df = load_data()
    st.success(f"Dataset loaded successfully! Total records: {len(df):,}")
except FileNotFoundError:
    st.error("Error: Could not find 'starwars_filtered.csv'. Please ensure the data file exists.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Year range filter
year_min, year_max = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

# Character type filter
character_types = ['All'] + sorted(df['character_type'].unique().tolist())
selected_character = st.sidebar.selectbox("Character Type", character_types)

# Condition filter
conditions = ['All'] + sorted(df['condition'].unique().tolist())
selected_condition = st.sidebar.selectbox("Condition", conditions)

# Authenticity filter
authenticity_options = ['All', 'Graded/Certified', 'Not Graded']
selected_authenticity = st.sidebar.selectbox("Authenticity", authenticity_options)

# Apply filters
df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if selected_character != 'All':
    df_filtered = df_filtered[df_filtered['character_type'] == selected_character]
if selected_condition != 'All':
    df_filtered = df_filtered[df_filtered['condition'] == selected_condition]
if selected_authenticity != 'All':
    auth_val = 1 if selected_authenticity == 'Graded/Certified' else 0
    df_filtered = df_filtered[df_filtered['authenticity_n'] == auth_val]

# Display filtered data metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered Records", f"{len(df_filtered):,}")
with col2:
    st.metric("Unique Figures", df_filtered['figure'].nunique())
with col3:
    st.metric("Avg Price", f"${df_filtered['selling_price'].mean():.2f}")
with col4:
    st.metric("Median Price", f"${df_filtered['selling_price'].median():.2f}")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Price Analysis",
    "Character Types",
    "Time Trends",
    "Top Figures",
    "Statistical Analysis"
])

# TAB 1: Overview
with tab1:
    st.header("Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Date Range:** {df['year'].min()} - {df['year'].max()}")
        st.write(f"**Unique Figures:** {df['figure'].nunique()}")
        st.write(f"**Character Types:** {len(df['character_type'].unique())}")
        st.write(f"**Conditions:** {', '.join(df['condition'].unique())}")

        st.subheader("Basic Statistics")
        st.dataframe(df_filtered.describe(), use_container_width=True)

    with col2:
        st.subheader("Authenticity Distribution")
        auth_dist = df_filtered['authenticity_n'].value_counts()
        auth_dist.index = ['Not Graded', 'Graded/Certified']

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#3498db', '#e74c3c']
        ax.pie(auth_dist.values, labels=auth_dist.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Grading Distribution')
        st.pyplot(fig)
        plt.close()

        st.subheader("Sample Data")
        st.dataframe(df_filtered.head(10), use_container_width=True)

# TAB 2: Price Analysis
with tab2:
    st.header("Price Distribution Analysis")

    # Price distributions by 4 segments
    st.subheader("Price Distribution by Condition and Grading/Certification")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Price Distribution by Condition and Grading/Certification', fontsize=16, fontweight='bold')

    combinations = [
        ('moc_figure', 1, 'MOC & Graded/Certified', axes[0, 0]),
        ('moc_figure', 0, 'MOC & Not Graded', axes[0, 1]),
        ('loose_figure', 1, 'Loose & Graded/Certified', axes[1, 0]),
        ('loose_figure', 0, 'Loose & Not Graded', axes[1, 1])
    ]

    for condition, auth, title, ax in combinations:
        data = df_filtered[(df_filtered['condition'] == condition) & (df_filtered['authenticity_n'] == auth)]['selling_price']

        if len(data) > 0:
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('Selling Price ($)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')

            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: ${data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: ${data.median():.2f}')

            ax.text(0.98, 0.97, f'n = {len(data):,}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary statistics table
    st.subheader("Summary Statistics by Market Segment")

    summary_data = []
    combinations = [
        ('moc_figure', 1, 'MOC & Graded/Certified'),
        ('moc_figure', 0, 'MOC & Not Graded'),
        ('loose_figure', 1, 'Loose & Graded/Certified'),
        ('loose_figure', 0, 'Loose & Not Graded')
    ]

    for condition, auth, label in combinations:
        data = df_filtered[(df_filtered['condition'] == condition) & (df_filtered['authenticity_n'] == auth)]['selling_price']
        if len(data) > 0:
            summary_data.append({
                'Category': label,
                'Count': len(data),
                'Mean': f'${data.mean():.2f}',
                'Median': f'${data.median():.2f}',
                'Std Dev': f'${data.std():.2f}',
                'Min': f'${data.min():.2f}',
                'Max': f'${data.max():.2f}'
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    # Boxplots
    st.subheader("Boxplot Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**With Outliers**")
        fig, ax = plt.subplots(figsize=(12, 8))

        price_data = []
        tick_labels = []

        combinations = [
            ('moc_figure', 1, 'MOC &\nGraded'),
            ('moc_figure', 0, 'MOC &\nNot Graded'),
            ('loose_figure', 1, 'Loose &\nGraded'),
            ('loose_figure', 0, 'Loose &\nNot Graded')
        ]

        for condition, auth, label in combinations:
            data = df_filtered[(df_filtered['condition'] == condition) & (df_filtered['authenticity_n'] == auth)]['selling_price']
            if len(data) > 0:
                price_data.append(data)
                tick_labels.append(label)

        if price_data:
            bp = ax.boxplot(price_data, tick_labels=tick_labels, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=8, markeredgecolor='darkred'))

            colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Selling Price ($)', fontsize=13)
            ax.set_xlabel('Market Segment', fontsize=13)
            ax.set_title('Price Distribution (With Outliers)', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("**Without Outliers**")
        fig, ax = plt.subplots(figsize=(12, 8))

        price_data_no_outliers = []
        tick_labels = []

        for condition, auth, label in combinations:
            data = df_filtered[(df_filtered['condition'] == condition) & (df_filtered['authenticity_n'] == auth)]['selling_price']

            if len(data) > 0:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                data_filtered = data[(data >= lower_bound) & (data <= upper_bound)]
                price_data_no_outliers.append(data_filtered)
                tick_labels.append(label)

        if price_data_no_outliers:
            bp = ax.boxplot(price_data_no_outliers, tick_labels=tick_labels, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', markersize=8, markeredgecolor='darkred'),
                            showfliers=False)

            colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Selling Price ($)', fontsize=13)
            ax.set_xlabel('Market Segment', fontsize=13)
            ax.set_title('Price Distribution (Without Outliers)', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            st.pyplot(fig)
            plt.close()

# TAB 3: Character Types
with tab3:
    st.header("Character Type Analysis")

    char_sales = df_filtered.groupby('character_type').agg({
        'selling_price': ['mean', 'median', 'count']
    }).round(2)
    char_sales.columns = ['Mean Price', 'Median Price', 'Count']
    char_sales = char_sales.sort_values('Mean Price', ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Average Prices by Character Type")
        st.dataframe(char_sales, use_container_width=True)

    with col2:
        st.subheader("Price Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(char_sales.index))
        width = 0.35

        ax.bar(x - width/2, char_sales['Mean Price'], width, label='Mean Price', alpha=0.8)
        ax.bar(x + width/2, char_sales['Median Price'], width, label='Median Price', alpha=0.8)
        ax.set_xlabel('Character Type')
        ax.set_ylabel('Price ($)')
        ax.set_title('Average Selling Price by Character Type')
        ax.set_xticks(x)
        ax.set_xticklabels(char_sales.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for i, count in enumerate(char_sales['Count']):
            ax.text(i, 5, f'n={int(count)}', ha='center', va='bottom', fontsize=8, rotation=90)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# TAB 4: Time Trends
with tab4:
    st.header("Time Series Analysis")

    df_time_filtered = df_filtered[df_filtered['year'] >= 2015]

    st.subheader("Price Trends Over Time (2015-2025)")

    fig, ax = plt.subplots(figsize=(16, 8))

    combinations = [
        ('moc_figure', 1, 'MOC & Graded/Certified', 'o', '#e74c3c', 3),
        ('moc_figure', 0, 'MOC & Not Graded', 's', '#3498db', 2.5),
        ('loose_figure', 1, 'Loose & Graded/Certified', '^', '#f39c12', 2.5),
        ('loose_figure', 0, 'Loose & Not Graded', 'D', '#2ecc71', 2)
    ]

    for condition, auth, label, marker, color, linewidth in combinations:
        subset = df_time_filtered[(df_time_filtered['condition'] == condition) & (df_time_filtered['authenticity_n'] == auth)]
        if len(subset) > 0:
            yearly = subset.groupby('year')['selling_price'].agg(['mean', 'median', 'count']).reset_index()

            ax.plot(yearly['year'], yearly['mean'], marker=marker, linewidth=linewidth,
                    label=label, markersize=7, color=color, alpha=0.8)

    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Average Selling Price ($)', fontsize=13)
    ax.set_title('Star Wars Figure Prices Over Time by Condition & Grading Status (2015-2025)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2015, 2026, 1))
    ax.set_xticklabels([str(year) for year in range(2015, 2026, 1)], rotation=0)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Sales Volume Over Time (2015-2025)")

    fig, ax = plt.subplots(figsize=(16, 8))

    for condition, auth, label, marker, color, linewidth in combinations:
        subset = df_time_filtered[(df_time_filtered['condition'] == condition) & (df_time_filtered['authenticity_n'] == auth)]
        if len(subset) > 0:
            yearly_counts = subset.groupby('year').size().reset_index(name='count')

            ax.plot(yearly_counts['year'], yearly_counts['count'], marker=marker, linewidth=linewidth,
                    label=label, markersize=7, color=color, alpha=0.8)

    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Number of Sales', fontsize=13)
    ax.set_title('Sales Volume Over Time by Condition & Grading Status (2015-2025)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(2015, 2026, 1))
    ax.set_xticklabels([str(year) for year in range(2015, 2026, 1)], rotation=0)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# TAB 5: Top Figures
with tab5:
    st.header("Top Figures Analysis")

    tab5_1, tab5_2, tab5_3 = st.tabs(["Most Valuable", "Best Selling", "Overall Top 10"])

    with tab5_1:
        st.subheader("Top 10 Most Valuable Figures by Market Segment")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        combinations = [
            ('moc_figure', 1, 'MOC & Graded/Certified', axes[0, 0], '#e74c3c'),
            ('moc_figure', 0, 'MOC & Not Graded', axes[0, 1], '#3498db'),
            ('loose_figure', 1, 'Loose & Graded/Certified', axes[1, 0], '#f39c12'),
            ('loose_figure', 0, 'Loose & Not Graded', axes[1, 1], '#2ecc71')
        ]

        for condition, auth, title, ax, color in combinations:
            subset = df_filtered[(df_filtered['condition'] == condition) & (df_filtered['authenticity_n'] == auth)]

            if len(subset) > 0:
                figure_stats = subset.groupby('figure').agg({
                    'selling_price': ['mean', 'count']
                })
                figure_stats.columns = ['avg_price', 'count']
                figure_stats = figure_stats[figure_stats['count'] >= 5]
                top_10 = figure_stats.nlargest(10, 'avg_price')

                if len(top_10) > 0:
                    y_pos = np.arange(len(top_10))
                    ax.barh(y_pos, top_10['avg_price'], color=color, alpha=0.8, edgecolor='black')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_10.index, fontsize=9)
                    ax.set_xlabel('Average Selling Price ($)', fontsize=10)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)

                    for i, (idx, row) in enumerate(top_10.iterrows()):
                        ax.text(row['avg_price'] + 10, i, f"n={int(row['count'])}",
                                va='center', fontsize=8, color='darkgray')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        fig.suptitle('Top 10 Most Valuable Figures by Market Segment (min 5 sales)',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab5_2:
        st.subheader("Top 10 Best Selling Figures by Market Segment")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        combinations = [
            ('moc_figure', 1, 'MOC & Graded/Certified', axes[0, 0], '#e74c3c'),
            ('moc_figure', 0, 'MOC & Not Graded', axes[0, 1], '#3498db'),
            ('loose_figure', 1, 'Loose & Graded/Certified', axes[1, 0], '#f39c12'),
            ('loose_figure', 0, 'Loose & Not Graded', axes[1, 1], '#2ecc71')
        ]

        for condition, auth, title, ax, color in combinations:
            subset = df_filtered[(df_filtered['condition'] == condition) & (df_filtered['authenticity_n'] == auth)]

            if len(subset) > 0:
                figure_sales = subset.groupby('figure')['sales'].sum().sort_values(ascending=False).head(10)

                if len(figure_sales) > 0:
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
                    ax.barh(y_pos, plot_data['total_sales'], color=color, alpha=0.8, edgecolor='black')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(plot_data['figure'], fontsize=9)
                    ax.set_xlabel('Total Sales Volume', fontsize=10)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)

                    for i, (idx, row) in enumerate(plot_data.iterrows()):
                        ax.text(row['total_sales'] + max(plot_data['total_sales'])*0.02, i,
                                f"${row['avg_price']:.0f}",
                                va='center', fontsize=8, color='darkgreen', fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        fig.suptitle('Top 10 Best Selling Figures by Market Segment (by total sales volume)',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab5_3:
        st.subheader("Overall Top 10 Figures (All Conditions & Authenticity)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top 10 Most Valuable Figures")

            # Calculate average price across all conditions
            figure_stats = df_filtered.groupby('figure').agg({
                'selling_price': ['mean', 'count']
            })
            figure_stats.columns = ['avg_price', 'count']
            figure_stats = figure_stats[figure_stats['count'] >= 5]  # At least 5 sales
            top_10_price = figure_stats.nlargest(10, 'avg_price')

            if len(top_10_price) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(top_10_price))
                ax.barh(y_pos, top_10_price['avg_price'], color='#9b59b6', alpha=0.8, edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_10_price.index, fontsize=10)
                ax.set_xlabel('Average Selling Price ($)', fontsize=11)
                ax.set_title('Top 10 Most Valuable Figures (Overall)', fontsize=13, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)

                # Add count annotations
                for i, (idx, row) in enumerate(top_10_price.iterrows()):
                    ax.text(row['avg_price'] + max(top_10_price['avg_price'])*0.02, i,
                            f"${row['avg_price']:.0f} (n={int(row['count'])})",
                            va='center', fontsize=9, color='darkgray')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Show table
                st.dataframe(top_10_price.style.format({'avg_price': '${:.2f}', 'count': '{:.0f}'}),
                            use_container_width=True)
            else:
                st.info("No data available for most valuable figures")

        with col2:
            st.markdown("### Top 10 Best Selling Figures")

            # Calculate total sales across all conditions
            figure_sales = df_filtered.groupby('figure')['sales'].sum().sort_values(ascending=False).head(10)

            if len(figure_sales) > 0:
                # Get average price for each
                avg_prices = []
                for figure in figure_sales.index:
                    avg_price = df_filtered[df_filtered['figure'] == figure]['selling_price'].mean()
                    avg_prices.append(avg_price)

                plot_data = pd.DataFrame({
                    'figure': figure_sales.index,
                    'total_sales': figure_sales.values,
                    'avg_price': avg_prices
                })

                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(plot_data))
                ax.barh(y_pos, plot_data['total_sales'], color='#e67e22', alpha=0.8, edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(plot_data['figure'], fontsize=10)
                ax.set_xlabel('Total Sales Volume', fontsize=11)
                ax.set_title('Top 10 Best Selling Figures (Overall)', fontsize=13, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)

                # Add average price annotations
                for i, (idx, row) in enumerate(plot_data.iterrows()):
                    ax.text(row['total_sales'] + max(plot_data['total_sales'])*0.02, i,
                            f"{row['total_sales']:.0f} units (${row['avg_price']:.0f} avg)",
                            va='center', fontsize=9, color='darkgreen', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Show table
                st.dataframe(plot_data.style.format({'total_sales': '{:.1f}', 'avg_price': '${:.2f}'}),
                            use_container_width=True)
            else:
                st.info("No data available for best selling figures")

# TAB 6: Statistical Analysis
with tab6:
    st.header("Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Analysis")

        numeric_cols = ['authenticity_n', 'selling_price', 'sales', 'year']
        correlation = df_filtered[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(correlation, use_container_width=True)

    with col2:
        st.subheader("Hypothesis Testing")

        # T-test for condition
        if 'loose_figure' in df_filtered['condition'].values and 'moc_figure' in df_filtered['condition'].values:
            loose = df_filtered[df_filtered['condition'] == 'loose_figure']['selling_price']
            moc = df_filtered[df_filtered['condition'] == 'moc_figure']['selling_price']

            if len(loose) > 0 and len(moc) > 0:
                t_stat, p_value = stats.ttest_ind(loose, moc)

                st.markdown("**T-test (Loose vs MOC):**")
                st.write(f"t-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.4e}")
                st.write(f"Result: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} difference in prices")

        st.markdown("---")

        # T-test for grading status
        if 0 in df_filtered['authenticity_n'].values and 1 in df_filtered['authenticity_n'].values:
            non_graded = df_filtered[df_filtered['authenticity_n'] == 0]['selling_price']
            graded = df_filtered[df_filtered['authenticity_n'] == 1]['selling_price']

            if len(non_graded) > 0 and len(graded) > 0:
                t_stat, p_value = stats.ttest_ind(non_graded, graded)

                st.markdown("**T-test (Not Graded vs Graded/Certified):**")
                st.write(f"t-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.4e}")
                st.write(f"Result: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} difference in prices")

        st.markdown("---")

        st.subheader("Key Insights")

        if len(df_filtered) > 0:
            most_expensive = df_filtered.loc[df_filtered['selling_price'].idxmax()]
            st.write(f"**Most expensive sale:** {most_expensive['figure']}")
            st.write(f"Price: ${most_expensive['selling_price']:.2f} in {int(most_expensive['year'])}")

            st.markdown("**Average price by condition:**")
            for condition, price in df_filtered.groupby('condition')['selling_price'].mean().items():
                st.write(f"- {condition}: ${price:.2f}")

            st.markdown("**Average price by grading status:**")
            for auth, price in df_filtered.groupby('authenticity_n')['selling_price'].mean().items():
                auth_label = "Graded/Certified" if auth == 1 else "Not Graded"
                st.write(f"- {auth_label}: ${price:.2f}")

            st.markdown("**Top 5 most valuable character types:**")
            for i, (char_type, price) in enumerate(df_filtered.groupby('character_type')['selling_price'].mean().sort_values(ascending=False).head(5).items(), 1):
                st.write(f"{i}. {char_type}: ${price:.2f}")

# Footer
st.markdown("---")
st.markdown("Data analysis of Star Wars action figure sales from 2009-2025")
