import pandas as pd 
from sklearn.utils import resample

class InsuranceDatasetBalance:
    """
    Balanced an imbalanced dataset using Random Undersampling 
    """

    def __init__(self):
        self.dataset = None
        self.balanced_dataset = None

    def load_dataset(self):
        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Handling Class Imbalance/Undersampling/insurance_claim_imbalanced.csv')

    def undersample_dataset(self):
        # Create the balanced dataset using Random Undersampling 

        # Sparate majority class 
        majority = self.dataset[self.dataset["FiledClaim"]==0]

        # Saparate manority class 
        minority = self.dataset[self.dataset["FiledClaim"]==1]

        # Ramdomly reduced the majority class 
        majority_undersampled = resample(majority,replace=False,n_samples=len(minority),random_state=42)

        # Combined the both class 
        self.balanced_dataset = pd.concat([majority_undersampled,minority])


    def get_balanced_dataset(self):

        # Return the balanced dataset 
        return self.balanced_dataset

    def get_class_distribution(self):

        # Return the class distribution after undersampling 
        return self.balanced_dataset["FiledClaim"].value_counts() 

if __name__ == "__main__":
    balancer = InsuranceDatasetBalance() 
    balancer.load_dataset() 
    balancer.undersample_dataset()

    print("Balanced dataset:") 
    print(balancer.get_balanced_dataset())

    print("\nClass Distribution")
    print(balancer.get_class_distribution())


