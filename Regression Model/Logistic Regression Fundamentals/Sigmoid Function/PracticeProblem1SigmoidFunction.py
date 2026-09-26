import pandas as pd 
import math 


def load_dataset():
    """
    Load the student dataset
    """
    # Read the csv file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Sigmoid Function/student_pass_probability.csv')

def calculate_sigmoid(z):
    """
    Calculate the sigmoid function
    """
    # Apply the sigmoid Function 
    return 1 / (
        1 + math.exp(-z) 
    )

def display_probabilities(dataset):
    "Calculate and display the predicted probabilities." 

    # Assumed the value are learned 
    intercept = -6 

    coefficient = -1 

    for _,row in dataset.iterrows():

        # Read the input feature 
        hours = row["HoursStudied"]

        # Calculate the z 
        z = intercept + coefficient * hours

        # Calculate the probability 
        probability = calculate_sigmoid(z) 

        print(f"Hours of Studied:",{hours})

        print(f"Predicted Probability:",
              f"{probability:.4f}"
              )

def main():
    # Load the dataset 
    dataset = load_dataset() 

    # Calculate and display 
    # the probabilites
    display_probabilities(dataset) 

if __name__ == "__main__":
    main()     

