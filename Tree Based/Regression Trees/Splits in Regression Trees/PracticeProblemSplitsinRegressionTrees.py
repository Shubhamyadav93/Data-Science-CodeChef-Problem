import pandas as pd 
from sklearn.tree import DecisionTreeRegressor

class HousePriceTreeAnalyzer:
    """
    Train a Regression Tree and analyze its first split 
    """

    def __init__(self):
        self.dataset = None
        self.model = None
        self.feature_names = ["Area","Bedrooms","HouseAge","Price"]

    def load_dataset(self):
        # Load the csv file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Tree Based/Regression Trees/Splits in Regression Trees/house_price_regression_tree.csv') 
    
    def prepare_data(self):
        pass 

    