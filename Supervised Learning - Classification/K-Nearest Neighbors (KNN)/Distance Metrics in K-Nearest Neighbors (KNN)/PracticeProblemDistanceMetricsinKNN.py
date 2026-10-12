import pandas as pd 
from scipy.spatial.distance import euclidean,cityblock 

class KNNDistanceCalculator:
    "Calculate the distance metrics used in KNN" 

    def __init__(self):
        self.dataset = None

    def load_dataset(self):
        # Load the CSV file 

        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Supervised Learning - Classification/K-Nearest Neighbors (KNN)/Distance Metrics in K-Nearest Neighbors (KNN)/fruit_knn_distance.csv') 

    def prepare_data(self):

        # Return the training samples and new sample 
        X_train = self.dataset.iloc[:-1]

        X_new = self.dataset.iloc[-1]

        return X_train,X_new 

    def calculate_euclidean_distances(self,X_train,X_new):
        # Return the euclidean_distance 

        distance = [] 
        for _,row in X_train.iterrows():
            distance.append(euclidean(row,X_new)) 

        return distance 

    def calculate_manhattan_distances(self,X_train,X_new):
        # Return the Manhattan distance 
        
        distance = [] 
        for _,row in X_train.iterrows():
            distance.append(cityblock(row,X_new))

        return distance 

if __name__ == "__main__":
    calculator = KNNDistanceCalculator() 

    calculator.load_dataset() 

    X_train , X_new = calculator.prepare_data()

    euclidean_distance = calculator.calculate_euclidean_distances(X_train,X_new)

    
    manhattan_distances =  calculator.calculate_manhattan_distances(X_train,X_new) 

    print("Euclidean distance") 

    for distance in euclidean_distance:
        print(f"{distance:.3f}")

    print("Manhattan distance") 

    for distance in manhattan_distances:
        print(f"{distance:.3f}")


