import pandas as pd
from sklearn.linear_model import LinearRegression

#  Load the dataset 
dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Simple Linear Regression/OLS-Least Squares/employee_salary.csv')

# Store the target value 
X = dataset[["YearsExperience"]]

# Store the target value 
y = dataset['Salary']

# Crate the Linear Regression model 
model = LinearRegression()

# Train the model 
# Ordinary Least Squares (OLS) is automatically applied here 
model.fit(X,y)

# Display the learned intercept and slope 
print("Intercept (b0):",model.intercept_)
print("Slop (b0):",model.coef_[0])

# Predict salaries 
predictions = model.predict(X) 

# Display the first five predicted salaries 
print("\nFirst Five Predicted Salaries")
for salary in predictions[:5]:
    print(salary)

