import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 

def load_dataset():
    """
    Load the flower dataset
    """
    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Supervised Learning - Classification/K-Nearest Neighbors (KNN)/Choosing the Value of K in K-Nearest Neighbors/flower_knn.csv') 


def prepare_data(dataset):
    """
    Separate the input feature and target value 
    """ 

    # Select the input feature 
    X = dataset[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

    # Select the target class. 
    y = dataset["Species"]  


    return train_test_split(X,y,test_size=0.3,random_state=42) 


def evaluate_knn_model(X_train,X_test,y_train,y_test):
    """
    Train KNN models with different values of k 
    """

    for k in [1,3,5,7]:
        # Create the KNN model 
        model = KNeighborsClassifier(n_neighbors=k) 

        # Train the model 
        model.fit(X_train,y_train) 

        # Predict the test data 
        predictions = model.predict(X_test) 

        # Calculate the accuracy score 
        accuracy = accuracy_score(y_test,predictions) 
        
        print(f"Accuracy of K = {k}")

        print(f"{accuracy:.4f}\n") 


     
def main():
    # Load the dataset 
    dataset = load_dataset() 

    # Prepare the data 
    X_train , y_train,X_test,y_test = prepare_data(dataset) 

    # Evaluate the difference of K 
    evaluate_knn_model(X_train,y_train,X_test,y_test) 

if __name__ =="__main__":
    main() 

