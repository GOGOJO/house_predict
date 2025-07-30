import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

cali_data = pd.read_csv("housing.csv")

# print(cali_data.columns)

# thing we want to predict 
y = cali_data.median_house_value

#values we use to predict  y 
cali_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms','ocean_proximity']
X = cali_data[cali_features]

# data split for training vs validation data
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
