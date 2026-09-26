import pandas as pd 
import math

def load_dataset():
    """
    Load the student dataset
    """ 
    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Decision Boundary/student_pass_classification.csv')


def calculate_sigmoid(z):
    return 1 / (
        1 + math.exp(-z)
    )


def classify(probability):
    """
    Classify using the default decision boundry 
    """

    # Apply the decision boundry 
    if probability >= 0.5:
        return 1 
    
    return 0 

def display_prediction(dataset):
    """
    Calculate and display the predicted 
    probabilites and predicted classes    
    """

    # Assumed the values are learn 
    intercept = -6 
    coefficient = 1 

    for _,row in dataset.iterrows():

        # Read the input feature 
        hours = row["HoursStudied"]

        # Calculate the z 
        z = intercept + coefficient * hours 

        # Calculate the probability 
        probability = calculate_sigmoid(z) 

        # Determine the predicted class 
        predicted_class = classify(probability)

        print(f"Hours studied: {hours}")

        print(
            f"Predicted Probability:"
            f"{probability:.4f}"
        )

        print(
            f"Preicted Class:"
            f"{predicted_class}"
        )

def main():
    # Load the dataset 
    dataset = load_dataset() 

    display_prediction(dataset)

if __name__ == "__main__":
    main() 

