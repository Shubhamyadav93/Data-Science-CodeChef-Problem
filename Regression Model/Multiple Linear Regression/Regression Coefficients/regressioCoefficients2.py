import pandas as pd 
from sklearn.linear_model import LinearRegression

class EmployeeIncomePredictor:
    """
    Train a Multiple Linear Regression model 
    and interpret the learned coefficients
    """

    def __init__(self):
        self.dataset = None
        self.X = None 
        self.y = None

        self.model = None

    def load_dataset(self):

        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Multiple Linear Regression/Regression Coefficients/employee_income.csv')

    def prepare_data(self):

        # Store the input features 
        self.X = self.dataset[["Experience","EducationLevel","WorkingHours"]]
        self.y = self.dataset["AnnualIncome"]


    def train_model(self):

        # Create the Multiple Linear Regression model 

        self.model = LinearRegression()

        # Train the model 
        self.model.fit(self.X,self.y)

    def get_model_parameterts(self):

        # Return the learned intercept and coefficients 
        intercept = self.model.intercept_
        coefficients = self.model.coef_

        return intercept,coefficients

    def get_coefficient_effects(self):
        # Return whether each coefficient 
        # has a Positive or Negative effect 
        coefficients = self.model.coef_
        effects = []
        for coefficient in coefficients:
            effect = "Positive" if coefficient > 0 else "Negative"
            effects.append(effect)

        return effects


if __name__ == "__main__":
    predictor = EmployeeIncomePredictor()

    predictor.load_dataset()

    predictor.prepare_data()

    predictor.train_model()

    intercept,coefficients = predictor.get_model_parameterts()
    print("Intercept (b0):",intercept)

    print("\nFeature Coefficients:")

    effects = predictor.get_coefficient_effects()

    for feature,coefficient,effects in zip(predictor.X.columns,coefficients,effects):
        print(f"{feature}: {coefficients} ({effects})")


