import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load and clean data
file_path = '../datasets/autos.csv'
columns_to_keep = ['curb-weight', 'engine-size', 'horsepower', 'peak-rpm', 'city-mpg', 'price']
na_values = ["?", "NA", "N/A", "na", "n/a", "", " "]

df = pd.read_csv(file_path, usecols=columns_to_keep, na_values=na_values)

# Convert all columns to numeric to catch invalid entries
df = df.apply(pd.to_numeric, errors='coerce')

# Show missing values before dropping them
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Fill missing values with rounded column means
df_filled = df.fillna(df.mean(numeric_only=True).round())
print("Missing values after filling:")
print(df_filled.isnull().sum())

# === Train-test split ===
X = df_filled.drop(columns=['price'])  # predictors
y = df_filled['price']                  # target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# === Train linear regression model ===
model = LinearRegression()
model.fit(X_train, y_train)

# Display coefficients and intercept
intercept = model.intercept_
coefficients = model.coef_

print("\nLinear Regression Model Coefficients:")
print(f"Intercept (b0): {intercept:.3f}")
for feature, coef in zip(X_train.columns, coefficients):
    print(f"  {feature}: {coef:.3f}")

# Print linear regression equation
equation_terms = [f"({coef:.3f} * {feature})" for feature, coef in zip(X_train.columns, coefficients)]
equation = "price = " + f"{intercept:.3f} + " + " + ".join(equation_terms)
print("\nLinear regression equation:")
print(equation)

# === Predict and evaluate on test set ===
y_pred = model.predict(X_test)

n = X_test.shape[0]  # number of observations
p = X_test.shape[1]  # number of predictors

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
msd = np.mean(y_pred - y_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nModel performance on test set:")
print(f"R2: {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Signed Difference (MSD): {msd:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# === Exclude one predictor at a time, retrain and evaluate R2 and adjusted R2 ===
predictors = ['curb-weight', 'engine-size', 'horsepower', 'peak-rpm', 'city-mpg']
results = []

for exclude_feature in predictors:
    X_train_reduced = X_train.drop(columns=[exclude_feature])
    X_test_reduced = X_test.drop(columns=[exclude_feature])
    
    model_reduced = LinearRegression()
    model_reduced.fit(X_train_reduced, y_train)
    y_pred_reduced = model_reduced.predict(X_test_reduced)
    
    r2_reduced = r2_score(y_test, y_pred_reduced)
    
    n_reduced = X_test_reduced.shape[0]
    p_reduced = X_test_reduced.shape[1]
    adjusted_r2_reduced = 1 - (1 - r2_reduced) * (n_reduced - 1) / (n_reduced - p_reduced - 1)
    
    results.append((exclude_feature, r2_reduced, adjusted_r2_reduced))

print("\nR-Squared and Adjusted R-Squared values on test data after excluding one predictor at a time:")
print("--------------------------------------------------------------")
print(f"{'Excluded Predictor':<17} | {'R-Squared':>9} | {'Adjusted R-Squared':>18}")
print("-------------------|-----------|--------------------")
for feature, r2_val, adj_r2_val in results:
    print(f"{feature:<17} | {r2_val:9.4f} | {adj_r2_val:18.4f}")

# Plot in a 2x3 grid using filled data
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, col in enumerate(predictors):
    ax = axes[i // 3, i % 3]
    sns.scatterplot(x=df_filled[col], y=df_filled['price'], ax=ax)
    ax.set_title(f'Price vs {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Price')

# Hide unused subplot (the 6th one)
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()
