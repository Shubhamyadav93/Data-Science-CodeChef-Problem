import pandas as pd 
from sklearn.linear_model import LinearRegression

# Load the dataset features 
dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Multiple Linear Regression/Regression Coefficients/house_price_analysis.csv')

# Store the input features 
X = dataset[["Area","Bedrooms","HouseAge"]]

# Store the target value 
y = dataset["Price"]

# Create the Multiple Linear Regression model 
model = LinearRegression()

# Train the model 
model.fit(X,y)

# Display the learned intercept 
print("\nFeature Coefficients:")

for feature,coefficient in zip(X.columns,model.coef_):
    effect = "Positive" if coefficient > 0 else "Negative"

    print(f"{feature}:{coefficient} ({effect})")



