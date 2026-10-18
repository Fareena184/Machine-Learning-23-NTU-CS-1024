import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("income_evaluation.csv")
df.columns = df.columns.str.strip()

df["income"] = df["income"].str.strip()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("income_>50K", axis=1)
y = df["income_>50K"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svc = SVC(random_state=101)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("Cross Validation Score:", cross_val_score(svc, X_train, y_train, cv=5).mean())
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))