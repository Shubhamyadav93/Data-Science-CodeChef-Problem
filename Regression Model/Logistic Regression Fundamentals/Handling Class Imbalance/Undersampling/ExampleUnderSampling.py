import pandas as pd 
from sklearn.utils import resample

def load_dataset():
    """
    Load the loan approval dataset 
    """

    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Handling Class Imbalance/Undersampling/loan_approval_imbalanced1.csv')


def undersampling_dataset(dataset):
    """
    Balanced the dataset using RandomUndersampling 
    """

    # Separate the mejority class 
    majority = dataset[dataset["LoanApproved"]==0]  

    # Separate the minority class 
    minority = dataset[dataset["LoanApproved"]==1]  

    # Randomly reduce the majority class 
    majority_undersampled = resample(majority,replace=False,n_samples=len(minority),random_state=42) 

    # Combined the both classes 
    balanced_dataset = pd.concat([majority_undersampled,minority]) 

    return balanced_dataset 

def display_class_distribution(original_dataset,balanced_dataset):
    """
    Display the class distribution before and after undersampling.  
    """

    print("Class Distribution Before Undersampling\n")

    print(original_dataset["LoanApproved"].value_counts())

    print("\nClass Distribution After Undersampling\n")

    print(balanced_dataset["LoanApproved"].value_counts()) 


def main():

    # Load the dataset 
    dataset = load_dataset() 

    # Balanced the dataset 
    balanced_dataset = (undersampling_dataset(dataset)) 

    # Display the class distribution 
    display_class_distribution(dataset,balanced_dataset) 


if __name__ == "__main__":
    main() 
