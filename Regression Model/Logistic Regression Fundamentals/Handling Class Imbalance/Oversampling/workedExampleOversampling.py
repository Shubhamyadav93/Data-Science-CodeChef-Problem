import pandas as pd 
from sklearn.utils import resample 

def load_dataset():
    """
    Load the loan approved dataset
    """

    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Handling Class Imbalance/Oversampling/loan_approval_imbalanced.csv')


def oversample_dataset(dataset):
    """
    Balanced the dataset using Random Oversampling. 
    """

    # Separate the majority class 
    majority = dataset[dataset["LoanApproved"] == 0 ]

    # Separate the minority class 
    minority = dataset[dataset["LoanApproved"] == 1] 

    # Duplicate the minority class 
    minority_oversampled = resample(minority,replace=True,n_samples = len(majority),random_state = 42) 

    # Combine the both classes 
    balanced_dataset = pd.concat([majority,minority_oversampled])

    return balanced_dataset


def display_class_distribution(original_dataset,balanced_dataset):
    """
    Display the class distribution befor and after oversampling 
    """

    print("Class Distribution Before oversampling:\n")

    print(original_dataset["LoanApproved"].value_counts()) 

    print("Class Distribution After oversampling:\n") 

    print(balanced_dataset["LoanApproved"].value_counts()) 



def main():

    # Load the dataset 
    dataset = load_dataset() 

    # Balanced the dataset 
    balanced_dataset = oversample_dataset(dataset) 

    # Display the class distribution 
    display_class_distribution(dataset,balanced_dataset)


if __name__ == "__main__":
    main()

