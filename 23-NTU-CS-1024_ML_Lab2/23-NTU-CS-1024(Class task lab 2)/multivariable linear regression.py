import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Admission_Predict.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove Serial Number column
df = df.drop(['Serial No.'], axis=1)

# Store target variable
y = df['Chance of Admit']

# Remove target variable from input features
df.drop("Chance of Admit", axis=1, inplace=True)

# Simple Linear Regression using GRE Score
simple_lr = LinearRegression()

simple_lr.fit(df[['GRE Score']], y)

# Plot actual data
plt.scatter(
    df['GRE Score'],
    y,
    color='green',
    label='Actual Data',
    alpha=0.5
)

# Plot regression line
plt.plot(
    df['GRE Score'],
    simple_lr.predict(df[['GRE Score']]),
    color='red',
    linewidth=3,
    label='Regression Line'
)

plt.xlabel('GRE Score')
plt.ylabel('Admission chance')
plt.title('GRE Score vs Admission chance')
plt.legend()
plt.show()