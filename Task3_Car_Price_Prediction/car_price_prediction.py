import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CodeAlpha Data Science Internship
# Task 3: Car Price Prediction
# ============================================================

print("Car Price Prediction")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("car data.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)


# ------------------------------------------------------------
# 2. Check Missing Values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. Dataset Information
# ------------------------------------------------------------

print("\nDataset Information:")
df.info()


# ------------------------------------------------------------
# 4. Prepare Data
# ------------------------------------------------------------

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]


# ------------------------------------------------------------
# 5. Separate Categorical and Numerical Columns
# ------------------------------------------------------------

categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_columns = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]


# ------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ------------------------------------------------------------
# 7. Create Machine Learning Model
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ------------------------------------------------------------
# 8. Split Dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))


# ------------------------------------------------------------
# 9. Train Model
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ------------------------------------------------------------
# 10. Prediction
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 11. Model Evaluation
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# ------------------------------------------------------------
# 12. Actual vs Predicted Prices
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))


# ------------------------------------------------------------
# 13. Predict Price of a New Car
# ------------------------------------------------------------

new_car = pd.DataFrame({
    "Car_Name": ["ritz"],
    "Year": [2015],
    "Present_Price": [5.5],
    "Driven_kms": [20000],
    "Fuel_Type": ["Petrol"],
    "Selling_type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0]
})

new_price = model.predict(new_car)

print("\nNew Car Prediction:")
print("Car Name       : ritz")
print("Year           : 2015")
print("Present Price  : 5.5 lakh")
print("Driven Kms     : 20000")
print("Fuel Type      : Petrol")
print("Selling Type   : Dealer")
print("Transmission   : Manual")
print("Owner          : 0")
print(f"Predicted Selling Price: {new_price[0]:.2f} lakh")


# ------------------------------------------------------------
# 14. Visualization - Actual vs Predicted
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. Visualization - Price Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(df["Selling_Price"], bins=20)

plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.title("Distribution of Car Selling Prices")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16. Final Message
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("Car Price Prediction Project Completed!")
print("=" * 60)