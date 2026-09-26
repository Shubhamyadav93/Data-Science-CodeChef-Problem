import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Simple Linear Regression/Cost Function/house_prices.csv')

# 2. Store the input feature (HouseArea) and target (HousePrice)
X = dataset[["HouseArea"]]
y = dataset["HousePrice"]

print("Total Record:",len(y))

# 3.Create a Linear Regression Model  
model = LinearRegression()

# 4. Train the model 
model.fit(X,y) 

# Predicted the House Price
prediction = model.predict(X) 

print("\nPredection Generation:")

# 6. Calculate the squared error for each prediction  
squared_error = (prediction-y)**2 

# 7.Calculate the cost 
n = len(y)
cost = np.sum(squared_error) / (2 * n) 

# 8. Display the Cost
print(f"\nCost:{cost:.2f}")


