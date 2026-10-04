import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("headbrain.csv")

X = data[["Head Size(cm^3)"]].values

y = data["Brain Weight(grams)"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

print("LINEAR REGRESSION")
print("-----------------")
print(f"R² Score : {linear_r2:.4f}")
print(f"RMSE     : {linear_rmse:.2f}")

poly2_model = Pipeline([
    (
        "poly",
        PolynomialFeatures(degree=2)
    ),
    (
        "linear",
        LinearRegression()
    )
])

poly2_model.fit(
    X_train,
    y_train
)

poly2_predictions = poly2_model.predict(
    X_test
)

poly2_r2 = r2_score(
    y_test,
    poly2_predictions
)

poly2_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        poly2_predictions
    )
)

poly3_model = Pipeline([
    (
        "poly",
        PolynomialFeatures(degree=3)
    ),
    (
        "linear",
        LinearRegression()
    )
])

poly3_model.fit(
    X_train,
    y_train
)

poly3_predictions = poly3_model.predict(
    X_test
)

poly3_r2 = r2_score(
    y_test,
    poly3_predictions
)

poly3_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        poly3_predictions
    )
)

poly4_model = Pipeline([
    (
        "poly",
        PolynomialFeatures(degree=4)
    ),
    (
        "linear",
        LinearRegression()
    )
])

poly4_model.fit(
    X_train,
    y_train
)

poly4_predictions = poly4_model.predict(
    X_test
)

poly4_r2 = r2_score(
    y_test,
    poly4_predictions
)

poly4_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        poly4_predictions
    )
)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression (Degree 2)",
        "Polynomial Regression (Degree 3)",
        "Polynomial Regression (Degree 4)"
    ],
    "R² Score": [
        linear_r2,
        poly2_r2,
        poly3_r2,
        poly4_r2
    ],
    "RMSE": [
        linear_rmse,
        poly2_rmse,
        poly3_rmse,
        poly4_rmse
    ]
})

print("\nMODEL COMPARISON")
print("----------------")
print(results)

X_curve = np.linspace(
    X.min(),
    X.max(),
    300
).reshape(-1, 1)

y_linear_curve = linear_model.predict(
    X_curve
)

y_poly2_curve = poly2_model.predict(
    X_curve
)

y_poly3_curve = poly3_model.predict(
    X_curve
)

y_poly4_curve = poly4_model.predict(
    X_curve
)

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X_curve,
    y_linear_curve,
    label="Linear Regression"
)

plt.plot(
    X_curve,
    y_poly2_curve,
    label="Polynomial Degree 2"
)

plt.plot(
    X_curve,
    y_poly3_curve,
    label="Polynomial Degree 3"
)

plt.plot(
    X_curve,
    y_poly4_curve,
    label="Polynomial Degree 4"
)

plt.xlabel("Head Size (cm³)")
plt.ylabel("Brain Weight (grams)")
plt.title("Linear and Polynomial Regression")
plt.legend()

plt.show()