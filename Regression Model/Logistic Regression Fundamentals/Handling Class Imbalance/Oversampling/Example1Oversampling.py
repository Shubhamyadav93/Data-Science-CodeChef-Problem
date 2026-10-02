import pandas as pd 
from sklearn.utils import resample 


class DiseaseDatasetBalance:
    """
    Balanced and Imbanced dataset using Random OverSampling.  
    """

    def __init__(self):
        self.dataset = None
        self.balanced_dataset = None

    def load_dataset(self):
        # Load the CSV file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Handling Class Imbalance/Oversampling/patient_disease_imbalanced.csv')

    def oversampling_dataset(self):

        # Create the balanced dataset using random oversampling 

        # Separate the majority class 
        majority = self.dataset[self.dataset["HasDisease"]==0] 

        # Separate the minority class 
        minority = self.dataset[self.dataset["HasDisease"]==1] 

        # Duplicate the minority class 

        minority_oversampled = resample(minority,replace=True,n_samples=len(majority),random_state=42)

        # Combined the both classes  
        self.balanced_dataset = pd.concat([majority,minority_oversampled])


    def get_balanced_dataset(self):

        # Return the balanced dataset 
        
        return self.balanced_dataset 

    def get_class_destribution(self):

        # Return the class distribution after oversampling 
        return self.balanced_dataset["HasDisease"].value_counts()

if __name__ == "__main__":

    balancer = DiseaseDatasetBalance() 

    balancer.load_dataset() 

    balancer.oversampling_dataset() 

    print("Balanced Dataset") 

    print(balancer.get_balanced_dataset())

    print("Class Distribution") 

    print(balancer.get_class_destribution())

