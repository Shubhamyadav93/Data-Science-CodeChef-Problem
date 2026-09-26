import pandas as pd 
from sklearn.linear_model import LinearRegression 

# 1. Load the Dataset 
dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Simple Linear Regression/Hypothesis in Linear Regression/car_prices.csv')

# 2.Select the input feature (CarAge) and target (SellingPrice)

X = dataset[["CarAge"]]
y = dataset["SellingPrice"]

# 3. Create a Linear Regression model 
model = LinearRegression()

# 4. Train the model using the dataset 
model.fit(X,y)

# 5. Get the learned intercept (b0) and slope (b1)
intercept = model.intercept_
slop = model.coef_[0]

# 6. Display the hypothesis equation 
print(f"Intercept (b0): {intercept:.2f}")
print(f"Slope (b1): {slop:.2f}")


print("\nHypothesis:")
print(f"ŷ = {intercept:.2f} + ({slop:.2f}) * x ")


# 7. Predict the selling price of a 5-year-old car 
car_age = pd.DataFrame({"CarAge":[5]})
prediceted_price = model.predict(car_age)

# Displcay the prediction 
print(f"\nCar Age: 5 years")
print(f"Predicted Selling Price: ₹{prediceted_price[0]:.2f}")



