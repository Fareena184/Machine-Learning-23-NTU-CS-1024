
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

df = pd.read_csv("creditcard.csv")

print("Dataset Preview:")
print(df.head())

print("\nClass Distribution:")
print(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest K:", grid_search.best_params_["knn__n_neighbors"])
print("Best Cross-Validation F1-Score:", grid_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1-Score:", f1_score(y_test, y_pred))
