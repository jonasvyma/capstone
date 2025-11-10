import streamlit as st
import matplotlib.pyplot as plt
st.set_page_config(page_title='Star Wars EDA', layout='wide')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_path = "../data/starwars/starwars_mocloose_sales_202510061222.csv"
df = pd.read_csv(csv_path, low_memory=False)

# Convert likely date/time columns
for col in df.columns:
    if any(k in col.lower() for k in ["date", "time", "timestamp"]):
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        except:
            pass

df.info()
df.head()



# ----- cell separator -----

missing = df.isna().sum().sort_values(ascending=False)
missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": (missing / len(df) * 100).round(2)
})
st.dataframe(missing_df)

print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")


# ----- cell separator -----

df.describe(include='all').T


# ----- cell separator -----

sns.set(style="whitegrid", palette="muted")

# Distribution of selling prices
plt.figure(figsize=(8,5))
sns.histplot(df['selling_price'], bins=40, kde=True)
plt.title("Distribution of Selling Prices")
st.pyplot(plt.gcf())

# Boxplot: Selling Price vs Condition
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="condition", y="selling_price")
plt.title("Selling Price by Condition")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Scatter: Authenticity vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="authenticity_n", y="selling_price", alpha=0.5)
plt.title("Authenticity vs Selling Price")
st.pyplot(plt.gcf())

# Top 10 Character Types by Sales
top_chars = df.groupby("character_type")["sales"].sum().nlargest(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_chars.index, y=top_chars.values)
plt.title("Top 10 Character Types by Total Sales")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())



# ----- cell separator -----

corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
st.pyplot(plt.gcf())


# ----- cell separator -----

if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    yearly = df.groupby("year")["selling_price"].median().dropna()
    plt.figure(figsize=(8,5))
    plt.plot(yearly.index, yearly.values, marker="o")
    plt.title("Median Selling Price by Year")
    plt.xlabel("Year")
    plt.ylabel("Median Price")
    st.pyplot(plt.gcf())


# ----- cell separator -----

plt.figure(figsize=(8,5))
sns.boxplot(x=df["selling_price"])
plt.title("Outlier Detection — Selling Price")
st.pyplot(plt.gcf())

# Optional: filter out extreme values
q_low, q_high = df["selling_price"].quantile([0.01, 0.99])
df_filtered = df[(df["selling_price"] >= q_low) & (df["selling_price"] <= q_high)]
print(f"Filtered dataset: {len(df_filtered):,} rows (1–99th percentile kept)")


# ----- cell separator -----

# Median price by condition
condition_summary = df.groupby("condition")["selling_price"].median().sort_values(ascending=False)
st.dataframe(condition_summary)

# Authenticity premium
auth_summary = df.groupby("authenticity_n")["selling_price"].mean().sort_values(ascending=False)
st.dataframe(auth_summary)


# ----- cell separator -----

df_filtered.to_csv("../data/starwars/starwars_mocloose_sales_clean.csv", index=False)


# ----- cell separator -----

