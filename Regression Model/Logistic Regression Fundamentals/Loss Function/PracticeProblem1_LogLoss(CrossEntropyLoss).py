import pandas as pd 
from sklearn.metrics import log_loss

class SpamDetectionEvaluator:
    """
    Calculate the Log Loss (Cross Entropy) for predicted probabilities.
    """

    def __init__(self):
        self.dataset = None
        self.y_true = None
        self.y_pred = None 

    def load_dataset(self):
        # Load the csv file 
        self.dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Loss Function/spam_log_loss.csv')

    def prepare_data(self):
        # Store the actual 
        # class label 
        self.y_true = self.dataset["ActualClass"] # these data are came list/array
        self.y_pred = self.dataset["PredictedProbability"] # these data are came in dataframe in the form of list/array 

    def calculate_log_loss(self):
        # Calculate and return the Log loss 
        return log_loss(self.y_true,self.y_pred)


if __name__ == "__main__":

    evaluator = SpamDetectionEvaluator() 

    evaluator.load_dataset() 

    evaluator.prepare_data() 

    loss = evaluator.calculate_log_loss() 

    print("Log Loss")
    print(f"{loss:.4f}")





    

    