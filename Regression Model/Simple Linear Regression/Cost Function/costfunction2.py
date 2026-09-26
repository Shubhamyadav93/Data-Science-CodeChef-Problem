import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression

class ApartmentRentCostCalculator:
    """
    Train a simple Linear Regression model and calculate the cost 
    its prediction.
    """

    def __init__(self):
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Simple Linear Regression/Cost Function/apartment_rent.csv')

    def prepare_data(self):
        # Store the input feature 
        self.X = self.dataset[["ApartmentSize"]]

        # Store the target value 
        self.y = self.dataset["MonthlyRent"]

    def train_model(self):
        # Create the Linear Regression model 
        self.model = LinearRegression()

        self.model.fit(self.X,self.y)

    def predict_rent(self):

        # predict the monthly rent 
        predictions = self.model.predict(self.X)
        return predictions

    def calculate_cost(self):
        # Get the rpredictions 
        predictions = self.predict_rent()

        # Calculate the squared errors 

        squared_errors = (predictions - self.y)**2

        # Calculate the cost 
        cost = np.sum(squared_errors)/(2*len(self.y))

        return cost 


if __name__ == "__main__":

    calculator = ApartmentRentCostCalculator()

    calculator.prepare_data()

    print("Total records: ",len(calculator.dataset))

    calculator.train_model()

    predictions = calculator.predict_rent() 

    print("\n Predictions generated.")

    print("\nCost:",calculator.calculate_cost())


