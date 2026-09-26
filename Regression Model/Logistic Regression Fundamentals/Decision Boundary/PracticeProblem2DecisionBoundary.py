import pandas as pd 
import math 

class LoanRepaymentClassifier:

    """
    Calculate probabilities using
    the Sigmoid Function and
    classify using a decision boundary.
    """

    def __init__(self):
        self.dataset = None
        self.intercept = -8 
        self.coefficient = 0.015 

    def load_dataset(self):
        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Decision Boundary/loan_repayment_prediction.csv')

    def calculate_sigmoid(self,z):
        return 1 / (
            1 + math.exp(-z) 
        )

    def predict_probabilities(self): 
        # Return the predicted probabilite 

        probabilities = [] 
        for _,row in self.dataset.iterrows():
             credit_score = row["CreditScore"] 

             z = self.intercept + self.coefficient * credit_score

             probability = self.calculate_sigmoid(z) 
             probabilities.append(probability) 

        return probabilities 


    def predict_classes(self):
        # Return the predicted classes using the decision boundry 

        probabilities = self.predict_probabilities() 

        predicted_classes = [] 

        for probability in probabilities:
            if probability >= 0.5:
                predicted_classes.append(1)
            else:
                predicted_classes.append(0) 

        return predicted_classes


if __name__ == "__main__":
    classifier = LoanRepaymentClassifier() 

    classifier.load_dataset()

    probabilities = classifier.predict_probabilities() 

    claseses = classifier.predict_classes() 

    print("Predicted Probabilities") 
    for probability in probabilities:
        print(f"{probability:.4f}")

    print("\nPrdicted Classes") 

    for predicted_class in claseses:
        print(predicted_class) 

