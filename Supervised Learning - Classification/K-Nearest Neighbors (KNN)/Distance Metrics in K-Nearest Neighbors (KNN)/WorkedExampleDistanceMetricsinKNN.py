import pandas as pd 
from scipy.spatial.distance import euclidean,cityblock

def load_dataset():
    "Load the dataset"

    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Supervised Learning - Classification/K-Nearest Neighbors (KNN)/Distance Metrics in K-Nearest Neighbors (KNN)/knn_distance_metrics.csv') 

def prepare_dataset(dataset):
    "Separate the training sample and new sample."

    # Taining samples 
    X_train = dataset.iloc[:-1]

    # New sample 
    X_new = dataset.iloc[-1]

    return X_train,X_new 

def calculate_distance(X_train,X_new):
    "Calculate Euclidian distance and Manhattan distances." 

    euclidean_distance = []
    manhattan_distance = [] 

    for _,row in X_train.iterrows():

        # Calculate the Euclidean Distance 
        euclidean_distance.append(euclidean(row,X_new)) 

        # Calculate the Manhattan Distance 
        manhattan_distance.append(cityblock(row,X_new))

    return euclidean_distance,manhattan_distance 


def display_distance(euclidean_distance,manhattan_distance):
    "Display the calculated distance " 
    print("Euclidean Distance") 

    for distance in euclidean_distance:
        print(f"{distance:.3f}")

    print("Manhattan Distance") 

    for distance in manhattan_distance:
        print(f"{distance:.3f}") 

def main():
    dataset = load_dataset() 

    # Prepare the dataset 
    X_train,X_new = prepare_dataset(dataset)

    # Calculate distance 
    euclidean_distances ,manhattan_distance = calculate_distance(X_train,X_new) 

    # Display the result 
    display_distance(euclidean_distances,manhattan_distance)

if __name__ == "__main__":
    main() 
