import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset 
dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Multiple Linear Regression/Model Diagnostics/student_exam_scores.csv')

# Store the input features 
X = dataset[["StudyHours","Attendance","AssignmentsCompleted"]]

# Store the target value 
y = dataset['ExamScore']

# Create the multiple Linear Regression 
model = LinearRegression() 

# Train the model 
model.fit(X,y)

# Predict the exam score 
predictions = model.predict(X)

# Calculate the residuals 
residuals = y - predictions

# Display the five residuals 
print("Five Residuals")
for residual in residuals[:5]:
    print(residual) 

# Calculate the R2 Score 
r2 = r2_score(y,predictions)

# Calculate the adjusted R2 
n = len(y) # idar X bhi o sakta hai 
p = X.shape[1]

adjusted_r2 = 1 - ((1 - r2)*(n-1)/(n-p-1))

# Display the evalution metrics 
print("\nR2 Score :",r2)
print("Adjusted R2 :",adjusted_r2)


