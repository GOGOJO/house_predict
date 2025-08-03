import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load data
cali_data = pd.read_csv("housing.csv")

# Target variable
y = cali_data.median_house_value

# Features (including a categorical one)
cali_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'ocean_proximity']
X = pd.get_dummies(cali_data[cali_features])  # One-hot encode 'ocean_proximity'

# Split into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Function to get MAE for a given tree size
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Evaluate the model with various tree sizes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t Mean Absolute Error: %d" % (max_leaf_nodes, my_mae))
