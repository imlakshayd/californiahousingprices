import os
import sys


import numpy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn


matplotlib.use("TkAgg")  # This is used to show a plot in another window


directory = "data/housing.csv"


pd.options.display.max_rows = 100


df = pd.read_csv(directory)


df = df.dropna()


long = df["longitude"].to_numpy() # A measure of how far west a house is. The higher value of this parameter means farther west. The California longitude value ranges from: 114° 8' W to 124°
lat = df["latitude"].to_numpy() # A measure of how far north a house is. The higher value of this parameter means farther north. The California Latitude value ranges from: 32° 30' N to 42° N
age = df["housing_median_age"].to_numpy() # Median age of a house within a block (a block has population of around 600 to 3000 people). The lower number for this parameter means the building is newer.
t_rooms = df["total_rooms"].to_numpy() # Total number of rooms among all houses within a block.
t_bedrooms = df["total_bedrooms"].to_numpy() # Total number of bedrooms among all houses within a block.
pop = df["population"].to_numpy() # Total number of people residing within a block.
house = df["households"].to_numpy() # Total number of households, a group of people residing within a home unit, for a block.
income = df["median_income"].to_numpy() # Median income for households within a block of houses (measured in tens of thousands of US Dollars)
value = df["median_house_value"].to_numpy() # Median house value for households within a block (measured in US Dollars) *** This is also the target ***
#prox = df["ocean_proximity"].to_numpy() #Location of the house w.r.t ocean/sea. Ocean_proximity indicating (very roughly) whether each block group is near the ocean, near the Bay area, inland or on an island. This parameter will allows us include and interpret the categorical variable while regressing the dataset.


#Normalization


mu_long = np.mean(long)
mu_lat = np.mean(lat)
mu_age = np.mean(age)
mu_t_rooms = np.mean(t_rooms)
mu_t_bedrooms = np.mean(t_bedrooms)
mu_pop = np.mean(pop)
mu_house = np.mean(house)
mu_income = np.mean(income)
mu_value = np.mean(value)


#Note i'm calling and doing the steps the same so later make an array of everything and use that to normalize it all on one go something like below
# mins = np.min(X, axis=0)
# maxs = np.max(X, axis=0)
# mus  = np.mean(X, axis=0)
# X_norm = (X – mus)/(maxs – mins)


long = (long - mu_long)/(max(long) - min(long))
lat = (lat - mu_lat)/(max(lat) - min(lat))
age = (age - mu_age)/(max(age) - min(age))
t_rooms = (t_rooms - mu_t_rooms)/(max(t_rooms) - min(t_rooms))
t_bedrooms = (t_bedrooms - mu_t_bedrooms)/(max(t_bedrooms) - min(t_bedrooms))
pop = (pop - mu_pop)/(max(pop) - min(pop))
house = (house - mu_house)/(max(house) - min(house))
income = (income - mu_income)/(max(income) - min(income))
value = (value - mu_value)/(max(value) - min(value))


X = np.stack((long, lat, age, t_rooms, t_bedrooms, pop, house, income), axis=1)


# Shuffle the data
indices = np.arange(X.shape[0])
np.random.seed(42)  # for reproducibility
np.random.shuffle(indices)


X = X[indices]
value = value[indices]


# 80% train, 20% test
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = value[:split], value[split:]


missing_values_per_column = df.isnull().mean()
print("Missing values per column:")
print(missing_values_per_column)


def mean_squared_error(y_true, y_pred):
   return np.mean((y_true - y_pred)**2)




print("--- DataFrame ---")
print(df.to_string())


def multiple_linear_regression_cl(X, y):


   m, n = X.shape # M is number of data points, while n is the number of features i.e. long, lat ...


   X_b = np.c_[np.ones((m, 1)), X] # np.ones to create and column which is filled with 1 which is "m" long and is then added to the orginal matrix of x using np.c_


   X_bt = numpy.transpose(X_b) # Transposing the array so we can calculate theta for each feature


   theta = np.linalg.inv(X_bt @ X_b) @ X_bt @ y # Equation for theta which is our weights for each feature


   return theta


theta_cf = multiple_linear_regression_cl(X_train, y_train)


print("Learned parameters (intercept first):", theta_cf)


def multiple_linear_regression_gd(X, y, lr=0.1, n_iters=1000):
   m, n = X.shape
   X_b = np.c_[np.ones((m,1)), X]
   theta = np.zeros(n+1)  # Making it so each feature has a weight of 0 to begin with and using gradient descent algorithm it can tweak it to the optimal weights the plus 1 is for the bias
   costs = []


   for i in range(n_iters):


       preds = X_b @ theta # y = X @ theta just matrix multiplication


       error = preds - y


       cost = np.mean(error**2)


       costs.append(cost)


       grads = (2/m) * (X_b.T @ error)


       theta -= lr * grads


   return theta, costs


theta_gd, costs = multiple_linear_regression_gd(X_train, y_train, lr=0.05, n_iters=2000)


print("GD parameters:", theta_gd)


m, n = X.shape


X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
pred_cf = X_test_b @ theta_cf
pred_gd = X_test_b @ theta_gd




mse_cf = mean_squared_error(y_test, pred_cf)
mse_gd = mean_squared_error(y_test, pred_gd)


print("Test MSE (Closed-form):", mse_cf)
print("Test MSE (GD):", mse_gd)


residuals_cf = y_test - pred_cf
residuals_gd = y_test - pred_gd


plt.figure()
plt.hist(residuals_cf, bins=50, alpha=0.5, label="Closed-form")
plt.hist(residuals_gd, bins=50, alpha=0.5, label="Gradient Descent")
plt.legend()
plt.title("Residual Distribution")
plt.xlabel("Residual (y - y_hat)")
plt.ylabel("Frequency")


plt.figure()
plt.scatter(y_test, pred_cf, alpha=0.5, label="Closed-form")
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
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent Convergence')
plt.show()

