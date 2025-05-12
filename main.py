import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

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

def multiple_linear_regression(long, lat, age, t_rooms, t_bedrooms, pop, house, income, value):


missing_values_per_column = df.isnull().mean()
print("Missing values per column:")
print(missing_values_per_column)

print("--- DataFrame ---")
print(df.to_string)