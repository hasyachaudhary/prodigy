# prodigy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/kc_house_data.csv')

df.head()

#hear we are gonna pridict house prise based on lotarea and bedroom
print(df.columns)
print(df.shape)

df.info()

df = df[['price', 'bedrooms', 'bathrooms', 'sqft_living','floors','view', 'grade',
       'sqft_above', 'sqft_basement', 'sqft_living15']]
print(df.describe())

df['sqft_above'] = df['sqft_above'].fillna(df['sqft_above'].mean())
print(df.isna().sum())

# Heatmap for better visualization of correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

X = df[['price', 'bedrooms', 'bathrooms', 'sqft_living','floors','view', 'grade',
       'sqft_above', 'sqft_basement', 'sqft_living15']]
y = df['price']

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# the coefficients
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# predicted vs actual values plot
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# minor residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Prices')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
