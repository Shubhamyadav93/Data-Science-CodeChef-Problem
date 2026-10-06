import pandas as pd 
from sklearn.tree import DecisionTreeRegressor

def load_dataset():
    """
    Load the house dataset. 
    """
    # Read the csv file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Tree Based/Regression Trees/Splits in Regression Trees/house_price_tree.csv') 

def prepare_data(dataset):
    """
    Separate the input features and target values 
    """
    # Select the input features 
    X = dataset[["Area","Bedrooms","HouseAge"]] 

    # Select the target values 
    y = dataset["Price"] 

    return X,y 

def train_model(X,y):
    """
    Train the Regression Tree 
    """ 
    # Create the Regression Tree 
    model = DecisionTreeRegressor(random_state=42,max_depth=3) 
    # Train the model 
    model.fit(X,y) 

    return model

def display_tree_information(model,X):
    """
    Display the first split and predicted price
    """

    feature_names = ["Area","Bedrooms","HouseAge"]

    # Get the first split feature
    feature_index = model.tree_.feature[0]

    # Get the split threshold 
    threshold = model.tree_.threshold[0] 

    print("First split feature:")

    print(feature_names[feature_index])

    print("\nSplit Threshold")

    print(f"{threshold:.2f}")

    # Predict the house prices
    predictions = model.predict(X)

    print("\nPrediction Priceses")

    for price in predictions:
        print(f"{price:.2f}")


def main():

    # Load the dataset 
    dataset = load_dataset()

    # Prepare the data 
    X,y = prepare_data(dataset) 

    # Train the model 
    model = train_model(X,y) 

    # Display tree information
    display_tree_information(model,X)


if __name__ == "__main__":
    main() 

