# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a capstone data analysis project focused on Star Wars action figures sales data. The project analyzes pricing trends, character types, conditions (loose vs MOC - Mint on Card), and authenticity across years (2009-2025). The analysis uses Python data science tools and is presented through multiple formats: Jupyter notebooks, Python scripts, and an interactive Streamlit dashboard.

## Data Architecture

### Data Sources
- **Primary dataset**: `claude/starwars_filtered.csv` (51,028 records)
- **Raw data location**: `data/starwars/` contains source CSV files:
  - `loose_figures.csv` and `loose_figures3.csv` - Loose figure sales data
  - `moc_figures.csv` and `moc_data - data_prep.csv` - Mint on Card figure data
  - Combined/processed versions with timestamps

### Data Schema
The filtered dataset contains 7 columns:
- `figure` (string) - Name of the Star Wars action figure
- `authenticity_n` (int) - Binary authenticity indicator (0=non-authentic, 1=authentic)
- `selling_price` (float) - Sale price in USD
- `sales` (float) - Sales volume/metric
- `condition` (string) - Figure condition: "loose_figure" or "moc_figure"
- `character_type` (string) - Character category: "droid", "alien/other", "jedi", "rebel", "empire/sith", "bounty_hunter", "human"
- `year` (int) - Year of sale (2009-2025)

### Data Processing
The SQL query in `notebooks/joinedstarwars.sql` shows how raw data is combined:
- MOC data is aggregated by figure/authenticity/condition/character_type/year/month
- Loose figure data is joined with raw sales prices
- Both sources are unioned into a single dataset

## Project Structure

### Analysis Files (in `output/` directory)
- **`sw_streamlit.py`** - Star Wars Streamlit dashboard (CURRENT PRIMARY FILE)
  - 6 tabs with interactive filters
  - Top Figures tab has 3 subtabs: Most Valuable, Best Selling, Overall Top 10
  - All visualizations use line graphs for time series (no bar charts for trends)
  - No COVID-19 annotations
  - Boxplots shown with and without outliers
- **`sw_analysis_notebook.ipynb`** - Star Wars Jupyter notebook (SYNCED WITH STREAMLIT)
  - Includes all visualizations from Streamlit app
  - Section 7: Overall Top 10 Figures analysis
  - All cells properly typed (markdown vs code)
- **Legacy files**: `claude.py`, `claude_notebook.ipynb`, `claude_streamlit.py` (older versions)

### Legacy/Iteration Files
- `starwars_eda_notebook*.ipynb` - Earlier analysis iterations
- `sw*.py` and `sw*.ipynb` - Previous Streamlit and notebook attempts

### Notebooks Directory
- `linear-regression.ipynb` - Statistical modeling
- `New_Complete_Dataset.ipynb` - Data preparation/cleaning
- `joinedstarwars.sql` - Data combination logic

### Data Directories
- `data/starwars/` - Star Wars figure data
- `data/pokemon/` - Pokemon card pricing data
- `images/` - Generated visualizations
- `pokemon_plots/` - Pokemon analysis visualizations

## Running the Project

### Star Wars Streamlit Dashboard (Primary Interface)
```bash
cd output
streamlit run sw_streamlit.py --server.port 8505
```
Access at: http://localhost:8505

**Current Features:**
- **6 main tabs**: Overview, Price Analysis, Character Types, Time Trends, Top Figures, Statistical Analysis
- **Top Figures has 3 subtabs**: Most Valuable (by segment), Best Selling (by segment), Overall Top 10 (aggregated)
- **Interactive filters**: year range slider, character type, condition, authenticity
- **Visualizations**: Line graphs for time series, boxplots with/without outliers
- **No COVID-19 annotations** in charts
- Statistical tests (t-tests)

### Star Wars Jupyter Notebook
```bash
cd output
jupyter notebook sw_analysis_notebook.ipynb
```
**Sections:**
1-6. Standard analysis (price, character types, time trends, boxplots)
7. **Overall Top 10 Figures** (new - aggregated across all segments)
8-10. Top selling by segment, grading impact, correlation, statistical tests

**Key Updates:**
- All time series use line graphs (not bar charts)
- Boxplots shown with and without outliers (side-by-side or sequential)
- No COVID-19 annotations
- Section numbering may need adjustment after cell insertions

## Key Dependencies

Core stack (from `sw_requirements.txt`):
- `streamlit` - Web dashboard framework
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations (used in all scripts)
- `scipy` - Statistical testing (t-tests, correlations)
- `numpy` - Numerical operations

## Analysis Components

### Streamlit App Architecture (`claude_streamlit.py`)
- **Data Loading**: `@st.cache_data` decorator on `load_data()` function for performance
- **Filter Logic**: Multi-dimensional filtering applied before passing to tabs (lines 82-94)
- **Tab Structure**: 6 independent tabs using `st.tabs()`, each with complete visualizations
- **Statistical Tests**: T-tests compare loose vs MOC and authentic vs non-authentic groups

### Visualization Patterns
All scripts follow consistent visualization approach:
- Seaborn "whitegrid" style with "husl" palette
- Default figure size: 12x6 inches
- Price distributions shown in both normal and log scales
- Bar charts with count annotations
- Time series with both mean and median trend lines

### Statistical Analysis
- **Correlation analysis**: Pearson correlation matrix for numeric variables
- **Hypothesis testing**: Independent t-tests for categorical comparisons (condition, authenticity)
- **Grouping operations**: Multi-level aggregations by character_type, year, figure name

## Important Implementation Notes

### File Path Dependencies
- All analysis scripts expect to run from the `claude/` directory
- The CSV file path is hardcoded as `'starwars_filtered.csv'` (relative path)
- Streamlit must be launched from `claude/` directory for data loading to work

### Matplotlib in Streamlit
- Use `st.pyplot(fig)` instead of `plt.show()`
- Always call `plt.close()` after `st.pyplot()` in loops to prevent memory leaks
- Figure size specifications are important for Streamlit's wide layout

### Data Filtering Pattern
Filters are applied sequentially in this order:
1. Year range (always applied)
2. Character type (if not "All")
3. Condition (if not "All")
4. Authenticity (if not "All")

This pattern ensures consistent behavior across all tabs.

## Common Analysis Queries

### Top N Analysis
- Most valuable figures: `df.groupby('figure')['selling_price'].mean().sort_values(ascending=False).head(N)`
- Most traded figures: `df['figure'].value_counts().head(N)`
- Character type rankings: `df.groupby('character_type')['selling_price'].mean().sort_values(ascending=False)`

### Time Series
- Yearly aggregation: `df.groupby('year').agg({'selling_price': ['mean', 'median', 'count']})`
- Price trend correlation: `df['selling_price'].corr(df['year'])`

### Categorical Comparisons
- Condition impact: T-test between loose_figure and moc_figure groups
- Authenticity impact: T-test between authenticity_n=0 and authenticity_n=1 groups
- Character type distribution: Value counts and pie charts

## Database Context

Original data sources from PostgreSQL schema `the_collectors`:
- `starwars_mocfigures` - MOC (Mint on Card) figures with monthly granularity
- `starwars_loosefigures` - Loose figures with individual sale records
- Combined table: `startwars_mocloose_sales` (note: typo in original SQL)

---

# Pokemon Card Pricing Analysis

## Project Overview

This is a parallel capstone data analysis project focused on Pokemon card pricing data. The project analyzes pricing trends across multiple condition types (Boxonly, Cib, Graded, Manualonly, New, Used), rarities, sets, and time periods (December 2020 - October 2025). The analysis uses the same Python data science stack and follows the same structure as the Star Wars project.

## Data Architecture

### Data Sources
- **Primary dataset**: `data/pokemon/final_dataset.csv` (1,274 records)
- **Data period**: Monthly pricing data from December 2020 to October 2025 (59 months)
- **Coverage**: 26 unique Pokemon cards tracked across time

### Data Schema
The dataset contains 13 columns (+ 3 derived):
- `Card ID` (int) - Unique identifier for each card
- `Card Name` (string) - Name of the Pokemon card (e.g., "Charizard Base Set")
- `Illustrator` (string) - Card artist (10 unique illustrators)
- `Rarity` (string) - Card rarity tier (12 types: Rare Holo, Shining Rare, Promo, etc.)
- `Release Date` (string) - Original card release date
- `Set Name` (string) - Pokemon TCG set (15 unique sets)
- `Date` (datetime) - Month of pricing data
- `Boxonly` (int) - Price for boxed/sealed products ($)
- `Cib` (int) - Complete in Box price ($)
- `Graded` (int) - Professionally graded card price ($)
- `Manualonly` (int) - Manual/instruction only price ($)
- `New` (int) - New condition price ($)
- `Used` (int) - Used condition price ($)
- **Derived columns**: `Year`, `Month`, `YearMonth`

### Data Characteristics
- **Zero values**: Significant (5-20% per condition) - indicates unavailable pricing data
- **Price ranges**: From $809 (Used min) to $23.8M (Used max - extreme outlier)
- **Top cards**: Charizard Base Set, Blastoise Base Set, Venusaur Base Set
- **Top rarities**: Rare Holo (447 records), Shining Rare (328 records)
- **Top sets**: Neo Destiny (328 records), Base Set (291 records)

## Project Structure

### Analysis Files (in `output/` directory)
- **`pk_streamlit.py`** - Pokemon Streamlit dashboard (CURRENT PRIMARY FILE)
  - **Booster Box rarity filtered out** (outliers removed from all analysis)
  - **Only 3 price columns used**: New, Used, Graded
  - 6 tabs with interactive filters matching Star Wars app structure
- **`pk_notebook.ipynb`** - Pokemon Jupyter notebook (SYNCED WITH STREAMLIT)
  - Same data filters as Streamlit (Booster Box excluded, 3 columns only)
  - All interpretation cells updated to reflect 3-column analysis
- **Legacy files**: `pokemon.py`, `pokemon_notebook.ipynb`, `pokemon_streamlit.py` (older versions)

### Generated Visualizations
The `pokemon_plots/` directory contains 10 PNG visualizations:
1. Price distributions (6-panel grid for all conditions)
2. Average price by condition (bar chart)
3. Price by rarity (top 8 rarities)
4. Price trends over time (multi-line time series)
5. Top 15 most valuable cards (horizontal bar)
6. Top 10 sets by price
7. Condition comparison boxplot (log scale)
8. Correlation heatmap (6x6 matrix)
9. Rarity distribution (pie chart, top 10)
10. Data volume over time (bar chart)

## Running the Pokemon Project

### Pokemon Streamlit Dashboard
```bash
cd output
streamlit run pk_streamlit.py --server.port 8504
```
Access at: http://localhost:8504

**IMPORTANT DATA FILTERS:**
- **Booster Box rarity EXCLUDED** (outliers removed)
- **Only 3 price columns**: New, Used, Graded (Boxonly, Cib, Manualonly NOT used)

**Dashboard features:**
- **6 tabs**: Overview, Price Analysis, Rarity Analysis, Time Trends, Top Cards, Statistical Analysis
- **Interactive filters**: Date range, rarity, set name, card name
- **Statistical tests**: ANOVA (3 conditions), t-test (New vs Used)
- All analysis reflects 3-column focus

### Python Script Analysis
```bash
cd claude
python pokemon.py
```
Generates 10 PNG visualizations in `pokemon_plots/` directory and outputs:
- Dataset overview statistics
- Price statistics by condition
- Rarity and set analysis
- Time series trends
- Top cards rankings
- Statistical tests (ANOVA, t-tests, correlations)

### Pokemon Jupyter Notebook
```bash
cd output
jupyter notebook pk_notebook.ipynb
```
**SYNCED WITH STREAMLIT:**
- Cell-3: Filters out Booster Box after data loading
- Cell-8: price_cols = ['New', 'Used', 'Graded'] (only 3)
- All interpretation cells updated for 3-column analysis
- Analysis structure matches Streamlit tabs

## Key Dependencies

Same core stack as Star Wars project:
- `streamlit` - Web dashboard framework
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `scipy` - Statistical testing (ANOVA, t-tests, correlations)
- `numpy` - Numerical operations

## Analysis Components

### Streamlit App Architecture (`pokemon_streamlit.py`)
- **Data Loading**: `@st.cache_data` decorator on `load_data()` for performance
- **Filter Logic**: Multi-dimensional filtering (date, rarity, set, card, condition)
- **Tab Structure**: 6 independent tabs using `st.tabs()`
- **Statistical Tests**: ANOVA for all conditions, t-test for New vs Used
- **Dynamic Metrics**: Real-time updates showing filtered record count, unique cards, date span

### Visualization Patterns
Consistent with Star Wars project approach:
- Seaborn "whitegrid" style with "husl" palette
- Default figure size: 12x6 inches (larger for multi-panel)
- Price distributions shown excluding zeros (non-zero filter)
- Currency formatting: `${x:,.0f}` for readability
- Log scale used for boxplots due to high price variance

### Statistical Analysis
- **Correlation analysis**: Pearson correlation matrix for 6 price conditions
  - Strongest correlation: Graded ↔ New (0.932)
  - Moderate correlations: Cib ↔ Graded (0.738), Cib ↔ New (0.724)
- **Hypothesis testing**:
  - ANOVA test across all 6 conditions (F=54.61, p<0.001) - significant differences
  - T-test New vs Used (t=-4.04, p<0.001) - significant difference
- **Time series analysis**: Correlation with Date to identify trends
  - Boxonly: Decreasing trend (corr = -0.266)
  - Others: Mostly stable (|corr| < 0.1)

## Important Implementation Notes

### File Path Dependencies
- All analysis scripts expect to run from the `claude/` directory
- The CSV file path is `'data/pokemon/final_dataset.csv'` (relative path)
- Streamlit must be launched from `claude/` directory for data loading to work

### Data Handling
- **Zero values**: Filtered out for most visualizations and statistics
- **Outliers**: Present but not removed (e.g., $23.8M Used price)
- **Missing data**: None (0 nulls across all columns)
- **Date conversion**: String to datetime with derived Year/Month columns

### Matplotlib in Streamlit
- Use `st.pyplot(fig)` instead of `plt.show()`
- Always call `plt.close()` after `st.pyplot()` in loops
- Currency formatter: `plt.FuncFormatter(lambda x, p: f'${x:,.0f}')`

## Common Analysis Queries

### Top N Analysis
- Most valuable cards: `df.groupby('Card Name')[price_cols].mean().mean(axis=1).sort_values(ascending=False).head(N)`
- Most valuable rarities: `df.groupby('Rarity')[price_cols].mean().mean(axis=1).sort_values(ascending=False)`
- Most valuable sets: `df.groupby('Set Name')[price_cols].mean().mean(axis=1).sort_values(ascending=False)`

### Time Series
- Monthly aggregation: `df.groupby('Date')[price_cols].mean()`
- Price trend correlation: `df[price_col].corr(df['Date'].astype(np.int64))`

### Categorical Comparisons
- Condition differences: ANOVA test across all 6 price columns
- New vs Used: Independent t-test
- Rarity price rankings: Group by rarity, calculate mean across conditions

## Key Insights from Analysis

### Most Valuable Items
1. **Top Card**: Booster Box 1st Edition ($1.38M average)
2. **Highest Single Price**: $23.9M (Used condition - extreme outlier)
3. **Top Rarity**: Booster Box ($1.38M avg)
4. **Top Set**: Neo Genesis ($895K avg)

### Price Patterns
- **Condition Rankings** (non-zero averages):
  1. Manualonly: $709K
  2. Used: $377K
  3. Boxonly: $260K
  4. Graded: $149K
  5. New: $89K
  6. Cib: $59K
- **Strong Correlations**: Graded-New (0.93), Cib-Graded (0.74), Cib-New (0.72)
- **Price Trends**: Boxonly decreasing, others stable

### Market Insights
- **Statistical Significance**: Conditions have significantly different prices (ANOVA p<0.001)
- **New vs Used**: Significant difference (t-test p<0.001)
- **Top Tracked Cards**: Base Set starters (Charizard, Blastoise, Venusaur) most frequently recorded
