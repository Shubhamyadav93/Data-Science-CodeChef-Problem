import pandas as pd 
from imblearn.over_sampling import SMOTE 

def load_dataset():
    """
    Load the loan approval dataset 
    """
    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Handling Class Imbalance/SMOTE (Synthetic Minority Over-sampling Technique)/loan_approval_imbalanced2.csv')

def apply_smote(dataset):
    """
    Balance the dataset using SMOTE. 
    """ 
    # Sparate the Inpute feature 
    X = dataset[["AnnualIncome","CreditScore"]]

    # Separate the target class 
    y = dataset["LoanApproved"] 

    # Create the SMOTE object 
    smote = SMOTE(random_state=42,k_neighbors=1)

    # Generate synthetic samples 
    X_resampled,y_resampled = smote.fit_resample(X,y) 

    # Create the balanced dataset 
    balanced_dataset = X_resampled.copy() 

    balanced_dataset["LoanApproved"]=y_resampled

    return balanced_dataset 


def display_class_classification(original_dataset,balanced_dataset):
    """
    Display the class distribution before and after SMOTE 
    """
    print("Class Distribution Before SMOTE:\n")
    print(original_dataset["LoanApproved"].value_counts())

    print("\nClass Distribution After SMOTE:\n")
    print(balanced_dataset["LoanApproved"].value_counts()) 

def main():

    # Load the dataset 
    dataset = load_dataset() 

    # Apply SMOTE 
    balanced_dataset = apply_smote(dataset) 

    # Display class Distribution 
    display_class_classification(dataset,balanced_dataset)


if __name__== "__main__":
    main() 


     
