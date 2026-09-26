import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class MedicalExpensesPredictor:
    """
        Train a Multiple Linear Regression model 
        and evaluate it using model diagnostics.

    """

    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.model = None

    def load_dataset(self):
        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Multiple Linear Regression/Model Diagnostics/medical_expenses.csv')

    def prepare_data(self):
        # Store the input features
        self.X = self.dataset[["Age", "BMI", "HospitalVisits"]]

        # Store the taregt value 
        self.y = self.dataset["MedicalExpenses"]

    def train_model(self):
        # Create the Multiple Linear Regression model 
        self.model = LinearRegression() 

        # Train the model 
        self.model.fit(self.X,self.y) 

    def predict_expenses(self):
        # Predicted the medical expenses 

        predictions = self.model.predict(self.X) 
        return predictions

    def calculate_residuals(self):
        # Calculate the residuals : actual - predicted 
        predictions = self.predict_expenses() 
        residuals = self.y - predictions 
        return residuals

    def calculate_r2_score(self):
        predictions = self.predict_expenses() 
        r2 = r2_score(self.y,predictions)
        return r2

    def calculate_adjusted_r2(self):
        # Calculate the Adjusted R2 
        # Formula: 1 - [(1 - R²) * (n - 1) / (n - p - 1)] 
        r2 = self.calculate_r2_score() 
        n = len(self.y)
        p = self.X.shape[1]
        adjusted_r2 = 1 - ((1-r2) * (n-1) / (n-p-1))
        return adjusted_r2 


if __name__ == "__main__":
    predictor = MedicalExpensesPredictor()
    predictor.load_dataset()
    predictor.prepare_data()
    predictor.train_model()

    residuals = predictor.calculate_residuals() 
    print("First Five Residuals:")
    for residual in residuals[:5]:
        print(residual)

    print("\nR² Score:")
    print(predictor.calculate_r2_score())

    print("\nAdjusted R²:") 
    print(predictor.calculate_adjusted_r2())


