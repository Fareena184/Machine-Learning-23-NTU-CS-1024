import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# Activity 3: Car Price Prediction
# ==========================================

# Load dataset
data = pd.read_csv("Car Price Prediction.csv")

# Display dataset
print("First 5 rows:")
print(data.head())

print("\nDataset Information:")
data.info()

print("\nColumn Names:")
print(data.columns.tolist())


# Remove rows with missing values
data = data.dropna()


# Select features mentioned in the activity
features = [
    "enginesize",
    "horsepower",
    "curbweight",
    "highwaympg"
]
# Target column

target = "price"

X = data[features]
y = data[target]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Linear Regression
# ==========================================

linear_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_pred)
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)

print("\nLinear Regression")
print("----------------------------")
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)

results = {}

for degree in [2, 3, 4]:

    polynomial_model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]
    )

    polynomial_model.fit(X_train, y_train)

    y_pred = polynomial_model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results[degree] = {
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\nPolynomial Regression - Degree {degree}")
    print("----------------------------")
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# ==========================================
# Compare Results
# ==========================================

print("\n================================")
print("Model Comparison")
print("================================")

print(
    f"Linear Regression     -> "
    f"RMSE: {linear_rmse:.2f}, "
    f"R2: {linear_r2:.4f}"
)

for degree, result in results.items():

    print(
        f"Polynomial Degree {degree} -> "
        f"RMSE: {result['RMSE']:.2f}, "
        f"R2: {result['R2']:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    linear_pred,
    label="Linear Regression"
)

plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")

plt.title("Actual vs Predicted Car Prices")

plt.legend()

plt.show()