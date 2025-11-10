#!/usr/bin/env python
# coding: utf-8

# In[5]:


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



# In[7]:


df.describe(include='all').T


# In[8]:


sns.set(style="whitegrid", palette="muted")

# Distribution of selling prices
plt.figure(figsize=(8,5))
sns.histplot(df['selling_price'], bins=40, kde=True)
plt.title("Distribution of Selling Prices")
plt.show()

# Boxplot: Selling Price vs Condition
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="condition", y="selling_price")
plt.title("Selling Price by Condition")
plt.xticks(rotation=45)
plt.show()

# Scatter: Authenticity vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="authenticity_n", y="selling_price", alpha=0.5)
plt.title("Authenticity vs Selling Price")
plt.show()

# Top 10 Character Types by Sales
top_chars = df.groupby("character_type")["sales"].sum().nlargest(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_chars.index, y=top_chars.values)
plt.title("Top 10 Character Types by Total Sales")
plt.xticks(rotation=45)
plt.show()



# In[9]:


corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[10]:


if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    yearly = df.groupby("year")["selling_price"].median().dropna()
    plt.figure(figsize=(8,5))
    plt.plot(yearly.index, yearly.values, marker="o")
    plt.title("Median Selling Price by Year")
    plt.xlabel("Year")
    plt.ylabel("Median Price")
    plt.show()


# In[11]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df["selling_price"])
plt.title("Outlier Detection — Selling Price")
plt.show()

# Optional: filter out extreme values
q_low, q_high = df["selling_price"].quantile([0.01, 0.99])
df_filtered = df[(df["selling_price"] >= q_low) & (df["selling_price"] <= q_high)]
print(f"Filtered dataset: {len(df_filtered):,} rows (1–99th percentile kept)")


# In[12]:


# Median price by condition
condition_summary = df.groupby("condition")["selling_price"].median().sort_values(ascending=False)
display(condition_summary)

# Authenticity premium
auth_summary = df.groupby("authenticity_n")["selling_price"].mean().sort_values(ascending=False)
display(auth_summary)


# In[13]:


df_filtered.to_csv("../data/starwars/starwars_mocloose_sales_clean.csv", index=False)


# In[ ]:




