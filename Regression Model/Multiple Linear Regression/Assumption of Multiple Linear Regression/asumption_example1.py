import pandas as pd 
from sklearn.linear_model import LinearRegression

class HousePricePrediction:
    """
        Train Multiple Linear Regression model 
        and practice house prices

    """

    def __init__(self):
        self.dataset = None
        self.X = None 
        self.y = None 

    def load_dataset(self):

        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Multiple Linear Regression/Assumption of Multiple Linear Regression/house_price_prediction.csv')


    def prepare_data(self):

        # Store the input features 
        self.X = self.dataset[['Area','Bedrooms','HouseAge']]

        # Store the target value 
        self.y = self.dataset['Price']

    def train_model(self):
        # Create the Multiple Linear Regression model 
        self.model = LinearRegression()

        # Train the model 
        self.model.fit(self.X,self.y)

    def get_model_parameters(self):

        # Return the learned intercept and coefficients 
        intercept = self.model.intercept_
        coefficients = self.model.coef_

        return intercept,coefficients

    def predict_prices(self):

        # Predict huouse prices
        prediction = self.model.predict(self.X)

        return prediction


if __name__ == "__main__":
    predictor = HousePricePrediction()
    predictor.load_dataset()
    predictor.prepare_data()
    predictor.train_model()

    intercept,coefficients = predictor.get_model_parameters()

    print("Intercept (b0):",intercept)

    print("\n Feature Coefficients:")
    for feature,coefficients in zip(predictor.X.columns,coefficients):
        print(f"{feature}:{coefficients}")

    predictions = predictor.predict_prices()

    print("\nFirst Five Predicted House Prices:")
    for price in predictions[:5]:
        print(price)



