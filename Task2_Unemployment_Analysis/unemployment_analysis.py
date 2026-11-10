
# ============================================================
# CodeAlpha Data Science Internship
# Task 2: Unemployment Analysis with Python
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

file_name = "Unemployment in India.csv"

df = pd.read_csv(file_name)

print("Unemployment Analysis with Python")
print("=" * 60)

# ------------------------------------------------------------
# 2. Clean Column Names
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

# Remove completely empty rows
df = df.dropna(how="all")

# Remove extra spaces from text columns
for column in ["Region", "Frequency", "Area"]:
    df[column] = df[column].astype(str).str.strip()

# Convert date column
df["Date"] = pd.to_datetime(
    df["Date"].astype(str).str.strip(),
    dayfirst=True,
    errors="coerce"
)

# Convert numeric columns
numeric_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing important values
df = df.dropna(
    subset=[
        "Region",
        "Date",
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)",
        "Area"
    ]
)

# ------------------------------------------------------------
# 3. Dataset Information
# ------------------------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

# ------------------------------------------------------------
# 4. Missing Values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 5. Basic Statistics
# ------------------------------------------------------------

print("\nBasic Statistics:")
print(df[numeric_columns].describe())

# ------------------------------------------------------------
# 6. Average Unemployment Rate
# ------------------------------------------------------------

average_unemployment = df["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Rate:")
print(f"{average_unemployment:.2f}%")

# ------------------------------------------------------------
# 7. Highest and Lowest Unemployment Records
# ------------------------------------------------------------

highest = df.loc[
    df["Estimated Unemployment Rate (%)"].idxmax()
]

lowest = df.loc[
    df["Estimated Unemployment Rate (%)"].idxmin()
]

print("\nHighest Unemployment Record:")
print(highest)

print("\nLowest Unemployment Record:")
print(lowest)

# ------------------------------------------------------------
# 8. Average Unemployment by Area
# ------------------------------------------------------------

area_average = (
    df.groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment by Area:")
print(area_average)

# ------------------------------------------------------------
# 9. Average Unemployment by Region
# ------------------------------------------------------------

region_average = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Regions by Average Unemployment:")
print(region_average.head(10))

# ------------------------------------------------------------
# 10. Monthly Unemployment Trend
# ------------------------------------------------------------

monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\nMonthly Unemployment Trend:")
print(monthly_unemployment)

# ------------------------------------------------------------
# 11. COVID-19 Period Analysis
# ------------------------------------------------------------

covid_period = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-06-30")
]

pre_covid_period = df[
    (df["Date"] >= "2019-05-01") &
    (df["Date"] < "2020-03-01")
]

covid_average = covid_period[
    "Estimated Unemployment Rate (%)"
].mean()

pre_covid_average = pre_covid_period[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nCOVID-19 Period Analysis:")
print(f"Average unemployment before March 2020: {pre_covid_average:.2f}%")
print(f"Average unemployment from March-June 2020: {covid_average:.2f}%")

# ------------------------------------------------------------
# 12. Maximum Unemployment During COVID-19 Period
# ------------------------------------------------------------

if not covid_period.empty:

    covid_max = covid_period.loc[
        covid_period["Estimated Unemployment Rate (%)"].idxmax()
    ]

    print("\nHighest Unemployment During March-June 2020:")
    print(covid_max)

# ------------------------------------------------------------
# 13. Visualization 1
# Monthly Unemployment Trend
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.title("Monthly Unemployment Rate Trend")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. Visualization 2
# Average Unemployment by Area
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

area_average.plot(kind="bar")

plt.title("Average Unemployment Rate by Area")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 15. Visualization 3
# Top 10 Regions by Average Unemployment
# ------------------------------------------------------------

top_10_regions = region_average.head(10)

plt.figure(figsize=(10, 6))

top_10_regions.plot(kind="bar")

plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 16. Visualization 4
# COVID-19 Unemployment Trend
# ------------------------------------------------------------

covid_monthly = (
    covid_period.groupby("Date")[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

plt.figure(figsize=(9, 5))

plt.plot(
    covid_monthly.index,
    covid_monthly.values,
    marker="o"
)

plt.title("Unemployment Rate During COVID-19 Period")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 17. Final Insights
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print(
    f"1. Overall average unemployment rate: "
    f"{average_unemployment:.2f}%"
)

print(
    f"2. Average unemployment before March 2020: "
    f"{pre_covid_average:.2f}%"
)

print(
    f"3. Average unemployment during March-June 2020: "
    f"{covid_average:.2f}%"
)

print(
    f"4. Highest recorded unemployment rate: "
    f"{highest['Estimated Unemployment Rate (%)']:.2f}% "
    f"in {highest['Region']} ({highest['Area']})"
)

print(
    f"5. Lowest recorded unemployment rate: "
    f"{lowest['Estimated Unemployment Rate (%)']:.2f}% "
    f"in {lowest['Region']} ({lowest['Area']})"
)

print("\n========================================")
print("Unemployment Analysis Project Completed!")
print("========================================")

