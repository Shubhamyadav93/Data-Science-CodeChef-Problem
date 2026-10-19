import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 

def load_dataset():
    """
    Load the patient dataset.
    """
    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Supervised Learning - Classification/K-Nearest Neighbors (KNN)/Curse of Dimensionality in K-Nearest Neighbors/patient_knn_dimensions.csv') 

def evaluate_knn(dataset):
    """
    Compare KNN using all features and selected features and selected features 
    """
    # Select all features 
    X_all = dataset[["Age","BloodPressure","Cholesterol","HeartRate","ExerciseHours","SleepHours"]]

    # Select important features 
    X_selected = dataset[["BloodPressure","Cholesterol","HeartRate"]] 

    # Select the target class 
    y = dataset["HasDisease"] 

    # Split the dataset for all features.
    X_train,X_test,y_train,y_test = train_test_split(X_all,y,test_size=0.3,random_state=42) 

    # Train KNN using selected features 
    model = KNeighborsClassifier(n_neighbors=3) 

    model.fit(X_train,y_train) 

    prediction = model.predict(X_test) 

    accuracy_all = accuracy_score(y_test,prediction) 

    # Split the dataset for selected features 
    X_train,X_test,y_train,y_test = train_test_split(X_selected,y,test_size=0.3,random_state=42)

    # Train KNN using selected features 
    model = KNeighborsClassifier(n_neighbors=3) 

    model.fit(X_train,y_train) 

    prediction = model.predict(X_test) 

    accuracy_selected = accuracy_score(y_test,prediction)

    return accuracy_all,accuracy_selected 


def main():
    # Load the dataset 
    dataset = load_dataset() 

    # Compare the both models 
    accuracy_all , accuracy_selected = evaluate_knn(dataset) 

    print("Accurucy using all Features:")
    print(f"{accuracy_all:.4f}") 

    print("\nAccurucy using selected Features:") 
    print(f"{accuracy_selected:.4f}") 

if __name__ == "__main__":
    main() 

