import os
import sys

import numpy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

matplotlib.use("TkAgg")  # This is used to show a plot in another window

# Data Ingestion
directory = "data/housing.csv"

pd.options.display.max_rows = 100

df = pd.read_csv(directory)


# Preprocessing - Removing all rows which don't have values
df = df.dropna()

# long = df["longitude"].to_numpy() # A measure of how far west a house is. The higher value of this parameter means farther west. The California longitude value ranges from: 114° 8' W to 124°
# lat = df["latitude"].to_numpy() # A measure of how far north a house is. The higher value of this parameter means farther north. The California Latitude value ranges from: 32° 30' N to 42° N
# age = df["housing_median_age"].to_numpy() # Median age of a house within a block (a block has population of around 600 to 3000 people). The lower number for this parameter means the building is newer.
# t_rooms = df["total_rooms"].to_numpy() # Total number of rooms among all houses within a block.
# t_bedrooms = df["total_bedrooms"].to_numpy() # Total number of bedrooms among all houses within a block.
# pop = df["population"].to_numpy() # Total number of people residing within a block.
# house = df["households"].to_numpy() # Total number of households, a group of people residing within a home unit, for a block.
# income = df["median_income"].to_numpy() # Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# value = df["median_house_value"].to_numpy() # Median house value for households within a block (measured in US Dollars) *** This is also the target ***
# prox = df["ocean_proximity"].to_numpy() #Location of the house w.r.t ocean/sea. Ocean_proximity indicating (very roughly) whether each block group is near the ocean, near the Bay area, inland or on an island. This parameter will allows us include and interpret the categorical variable while regressing the dataset.


# Feature Selection
features = df[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]].to_numpy()

value = df["median_house_value"].to_numpy()

# Standardization

mins = np.min(features, axis=0)
maxs = np.max(features, axis=0)
mus = np.mean(features, axis=0)

X = (features - mus)/(maxs - mins)

value = (value - np.mean(value))/(np.max(value) - np.min(value))

missing_values_per_column = df.isnull().mean()
print("Missing values per column:")
print(missing_values_per_column)

print("--- DataFrame ---")
print(df.to_string())

# Data splitting
# Shuffle the data
indices = np.arange(X.shape[0])
np.random.seed(42)
np.random.shuffle(indices)

X = X[indices]
value = value[indices]

# 80% train, 20% test
split = int(.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = value[:split], value[split:]

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def multiple_linear_regression_cl(X, y):
    d, n = X.shape  # d is number of data points, while n is the number of features i.e. long, lat ...

    X_b = np.c_[np.ones((d, 1)), X] # np.ones to create and column which is filled with 1 which is "d" long and is then added to the orginal matrix of x using np.c_

    X_bt = numpy.transpose(X_b) # Transposing the array so we can calculate theta for each feature

    theta = np.linalg.inv(X_bt @ X_b) @ X_bt @ y  # Equation for theta which is our weights for each feature

    return theta

theta_cf = multiple_linear_regression_cl(X_train, y_train)

print("Learned parameters (intercept first):", theta_cf)

def multiple_linear_regression_gd(X, y, alpha=0.1, iterations=1000):
    d, n = X.shape

    X_b = np.c_[np.ones((d, 1)), X]

    theta = np.zeros(n+1) # Making it so each feature has a weight of 0 to begin with and using gradient descent algorithm it can tweak it to the optimal weights the plus 1 is for the bias

    costs = []

    for i in range(iterations):

        preds = X_b @ theta # y = X @ theta just matrix multiplication

        error = preds - y

        cost = np.mean(error**2)

        costs.append(cost)

        X_bt = numpy.transpose(X_b)

        descent = (2/d) * (X_bt @ error)

        theta -= alpha * descent

    return theta, costs

theta_gd, costs = multiple_linear_regression_gd(X_train, y_train, alpha=0.05, iterations=5000)
print("GD parameters:", theta_gd)

d, n = X.shape

X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

pred_cf = X_test_b @ theta_cf

pred_gd = X_test_b @ theta_gd

mse_cf = mean_squared_error(y_test, pred_cf)
mse_gd = mean_squared_error(y_test, pred_gd)

residuals_cf = y_test - pred_cf
residuals_gd = y_test - pred_gd

plt.figure()
plt.hist(residuals_cf, bins=50, alpha=0.5, label="Closed-form")
plt.hist(residuals_gd, bins=50, alpha=0.5, label="Gradient Descent")
plt.legend()
plt.title("Residual Distribution")
plt.xlabel("Residual (actual - predictions)")
plt.ylabel("Frequency")

plt.figure()
plt.scatter(y_test, pred_cf, alpha=0.5, label="Closed-from")
plt.scatter(y_test, pred_gd, alpha=0.5, label="Gradient Descent")
plt.xlabel("True Normalized Value")
plt.ylabel("Predicted Normalized Value")
plt.title("Predictions vs. True Values")
plt.legend()

plt.figure()
plt.scatter(pred_cf, pred_gd, alpha=0.5)
plt.xlabel('Closed-form Prediction')
plt.ylabel('GD Prediction')
plt.title('GD vs. Closed-Form Predictions')

plt.figure()
for alpha in [0.005, 0.05, 0.5]:
    theta_gd, costs = multiple_linear_regression_gd(X_train, y_train, alpha, iterations=5000)
    plt.plot(costs, label=f"α = {alpha}")
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.show()