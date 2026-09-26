import pandas as pd 
import math 

class SigmoidProbabilityCalculator:
    """
    Calculate probabilities using the Sigmoid Function
    """ 

    def __init__(self):

        self.dataset = None
        self.intercept = -8 
        self.coefficient = 0.08 


    def load_dataset(self):
        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Sigmoid Function/patient_disease_probability.csv')



    def calculate_sigmoid(self,z):
        # Calculate the sigmoid Function 
        return 1 / (
            1 + math.exp(-z) 
        )

    def calculate_probabilities(self):
        # Calculate the predicted probabilites 
        probabilities = []
        for _ , row in self.dataset.iterrows(): # dataset me se BloodSugarLevel value ko nikalne ke liye humne ye loop chalaya 
            blood_sugar = row["BloodSugarLevel"]

            z = self.intercept + self.coefficient * blood_sugar 

            probability = self.calculate_sigmoid(z) 
            probabilities.append(probability)

        return probabilities 

if __name__ == "__main__":
    calculator = SigmoidProbabilityCalculator() 

    calculator.load_dataset() 
    probabolities = calculator.calculate_probabilities()

    print("Predicted Probababilites")
    
    for probability in probabolities:
        print(f"{probability:.4f}")

