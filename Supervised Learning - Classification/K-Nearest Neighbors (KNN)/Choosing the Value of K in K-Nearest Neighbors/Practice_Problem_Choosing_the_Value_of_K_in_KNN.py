import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNModelEvaluator: 
    """
    Evaluate KNN models using different value of k  
    """

    def __init__(self):
        self.dataset = None 

    def load_dataset(self):
        # Load the csv file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Supervised Learning - Classification/K-Nearest Neighbors (KNN)/Choosing the Value of K in K-Nearest Neighbors/patient_knn.csv') 

    def prepare_data(self):
        # Return the Training and testing the data 
        X = self.dataset[["Age","BloodPressure","Cholesterol","HeartRate"]]

        y = self.dataset["HasDisease"]

        return train_test_split(X,y,test_size=0.3,random_state=42)

    def evaluate_models(self,X_train,x_test,y_train,y_test):
        # Return the accuracy for each value of k 
        accuracies = {} 

        for k in [1,3,5,7]:
            # Create the KNN model 
            model = KNeighborsClassifier()

            # Train the model 
            model.fit(X_train,y_train) 

            # Predict the test data 
            predictions = model.predict(x_test) 

            accuracies[k] = accuracy_score(y_test,predictions) 

        return accuracies 

if __name__ == "__main__":
    evaluator = KNNModelEvaluator() 

    evaluator.load_dataset()

    X_train,x_test,y_train,y_test = evaluator.prepare_data() 

    accuracies = evaluator.evaluate_models(X_train,x_test,y_train,y_test) 

    for k, accuracy, in accuracies.items():
        print(f"Accuracy for K = {k}:") 
        print(f"{accuracy:.4f}\n") 




# --------------------------------------------------------------------------------------------------------------------------------------------
# Tasks to Perform
# --------------------------------------------------------------------------------------------------------------------------------------------

# Load the dataset.
# Separate the input features and target class.
# Split the dataset into training and testing sets.
# Train KNN classifiers using K = 1, 3, 5, and 7.
# Calculate the accuracy for each value of K.
# Return the accuracies.

# --------------------------------------------------------------------------------------------------------------------------------------------

