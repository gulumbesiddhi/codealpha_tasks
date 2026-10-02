
# ============================================================
# CodeAlpha Data Science Internship
# Task 1: Iris Flower Classification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("Iris.csv")

print("Iris Flower Classification")
print("=" * 50)

# ------------------------------------------------------------
# 2. Display Dataset
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ------------------------------------------------------------
# 3. Species Count
# ------------------------------------------------------------

print("\nSpecies Count:")
print(df["Species"].value_counts())

# ------------------------------------------------------------
# 4. Check Missing Values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 5. Prepare Features and Target
# ------------------------------------------------------------

X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
].values

y = df["Species"].values

# ------------------------------------------------------------
# 6. Split Dataset
# ------------------------------------------------------------

np.random.seed(42)

indices = np.arange(len(df))
np.random.shuffle(indices)

train_size = int(0.80 * len(df))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))

# ------------------------------------------------------------
# 7. K-Nearest Neighbors (KNN)
# ------------------------------------------------------------

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def knn_predict(X_train, y_train, X_test, k=5):

    predictions = []

    for test_point in X_test:

        distances = []

        for i, train_point in enumerate(X_train):

            distance = euclidean_distance(test_point, train_point)

            distances.append((distance, y_train[i]))

        distances.sort(key=lambda x: x[0])

        nearest_neighbors = distances[:k]

        votes = {}

        for distance, label in nearest_neighbors:

            if label not in votes:
                votes[label] = 0

            votes[label] += 1

        predicted_class = max(votes, key=votes.get)

        predictions.append(predicted_class)

    return np.array(predictions)


# ------------------------------------------------------------
# 8. Model Prediction
# ------------------------------------------------------------

y_pred = knn_predict(
    X_train,
    y_train,
    X_test,
    k=5
)

# ------------------------------------------------------------
# 9. Calculate Accuracy
# ------------------------------------------------------------

accuracy = np.mean(y_pred == y_test) * 100

print("\nModel Accuracy:")
print(f"{accuracy:.2f}%")

# ------------------------------------------------------------
# 10. Actual vs Predicted
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))

# ------------------------------------------------------------
# 11. Predict a New Flower
# ------------------------------------------------------------

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

prediction = knn_predict(
    X_train,
    y_train,
    new_flower,
    k=5
)

print("\nNew Flower Prediction:")
print("Sepal Length = 5.1 cm")
print("Sepal Width  = 3.5 cm")
print("Petal Length = 1.4 cm")
print("Petal Width  = 0.2 cm")
print("Predicted Species:", prediction[0])

# ------------------------------------------------------------
# 12. Visualization 1 - Species Count
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

df["Species"].value_counts().plot(kind="bar")

plt.title("Iris Flower Species Count")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 13. Visualization 2 - Sepal Length vs Sepal Width
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for species in df["Species"].unique():

    species_data = df[df["Species"] == species]

    plt.scatter(
        species_data["SepalLengthCm"],
        species_data["SepalWidthCm"],
        label=species
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. Visualization 3 - Petal Length vs Petal Width
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for species in df["Species"].unique():

    species_data = df[df["Species"] == species]

    plt.scatter(
        species_data["PetalLengthCm"],
        species_data["PetalWidthCm"],
        label=species
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.tight_layout()
plt.show()

print("\n========================================")
print("Iris Classification Project Completed!")
print("========================================")

