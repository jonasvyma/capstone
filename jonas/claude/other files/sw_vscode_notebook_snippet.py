# --- Quick EDA Starter in VSCode Notebook ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = r"/mnt/data/starwars_mocloose_sales_202510061222.csv"
df = pd.read_csv(csv_path, low_memory=False)

# Parse likely dates
for col in df.columns:
    if any(k in col.lower() for k in ["date", "time", "timestamp"]):
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        except Exception:
            pass

print("Shape:", df.shape)
display(df.head())

# Missingness
missing = df.isna().sum().sort_values(ascending=False)
display(pd.DataFrame({ "missing": missing, "missing_pct": (missing/len(df)*100).round(2) }))

# Numeric histogram example
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if num_cols:
    col = num_cols[0]
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col); plt.ylabel("Count")
    plt.show()

# Categorical bar example
cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
if cat_cols:
    col = cat_cols[0]
    vc = df[col].value_counts().head(20)
    plt.figure()
    vc.plot(kind="bar")
    plt.title(f"Top categories for {col}")
    plt.xlabel(col); plt.ylabel("Count")
    plt.show()
