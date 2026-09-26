import pandas as pd 
from sklearn.linear_model import LinearRegression

# Load data set
dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Multiple Linear Regression/Assumption of Multiple Linear Regression/student_performance.csv')

# Store the input features 

X = dataset[['StudyHours','Attendance','AssignmentsCompleted']]

# Store the target Value 
y = dataset['ExamScore']

# Create the multiple Linear Regression Model 
model = LinearRegression()

# train the model 
model.fit(X,y)

# Display the learn intercept 
print("Intercept (b0):",model.intercept_)

# Display the learn Coefficients 
print("\nFeature Coeffiencet")
for feature,cofficient in zip(X.columns,model.coef_):
    print(f'{feature}:{cofficient}')

# Predict Exam Score 
predictions = model.predict(X) 

# Display the first Five predicted scores 
print("\nFirst Price Predicted Scores:")
for score in  predictions[:5]:
    print(score)


# Intercept (b0): 17.77820874471078

# Feature Coeffiencet
# StudyHours:3.5052891396332857
# Attendance:0.4136107193229911
# AssignmentsCompleted:0.5306770098730595

# First Price Predicted Scores:
# 57.932299012693925
# 64.03631875881523
# 68.89950634696754
# 74.17630465444287



