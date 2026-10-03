import pandas as pd 
from imblearn.over_sampling import SMOTE

class LoanDefaultBalancer:
    "Balance and imbalanced dataset using SMOTE"

    def __init__(self):
        self.dataset = None
        self.balanced_dataset = None

    def load_dataset(self):

        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Handling Class Imbalance/SMOTE (Synthetic Minority Over-sampling Technique)/loan_default_imbalanced.csv')

    def apply_smote(self):

        # Create balanced dataset using SMOTE 
        # Separate input features 

        # Separate the input feature 
        X = self.dataset[["AnnualIncome","CreditScore"]]
        # Separete the target class 
        y = self.dataset["LoanDefault"]

        # Cretate the SMOTE object 
        smote = SMOTE(random_state=42,k_neighbors=1) 

        # Generate the synthetic samples 
        X_resampled,y_resampled = smote.fit_resample(X,y) 

        # Create the balanced dataset 
        self.balanced_dataset = X_resampled.copy() 

        self.balanced_dataset["LoanDefault"] = y_resampled


    def get_balanced_dataset(self):

        # Return the balance dataset 
        return self.balanced_dataset

    def get_class_distribution(self):
        # Return the class distribution after applying smote 
        return self.balanced_dataset["LoanDefault"].value_counts() 

if __name__ == "__main__":

    balancer = LoanDefaultBalancer() 

    balancer.load_dataset() 

    balancer.apply_smote() 

    print("Balanced dataset")
    print(balancer.get_balanced_dataset())

    print("\nClass Distribution")
    print(balancer.get_class_distribution()) 




