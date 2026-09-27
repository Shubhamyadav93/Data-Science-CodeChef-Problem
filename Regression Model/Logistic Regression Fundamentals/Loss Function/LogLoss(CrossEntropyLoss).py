import pandas as pd 
from sklearn.metrics import log_loss


def load_dataset():
    """
    Load the student dataset.
    """

    # Read the CSV file 
    return pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/Data Science CodeChef Problem/Regression Model/Logistic Regression Fundamentals/Loss Function/student_log_loss.csv')


def prepare_data(dataset):
    """
    Separate the actual class labels and predicted probabilites.
    """

    # Select the actual class labels 
    y_true = dataset["ActualClass"] 

    # Select the predicted probabilities 
    y_pred = dataset["PredictedProbability"]

    return y_true , y_pred 

def calculate_log_loss_value(y_true,y_pred):
    """
    Calculate the log loss 
    """

    # Calculate the log loss 
    return log_loss(y_true,y_pred)


def main():

    #  Load the dataset 
    dataset = load_dataset() 

    # Prepare the data 
    y_true,y_pred = prepare_data(dataset)

    # Calculate the Log loss 
    loss = calculate_log_loss_value(y_true,y_pred) 

    # Display the result 
    print("Log Loss:")
    print(f"{loss:.4f}")

if __name__ == "__main__":
    main()


# Log Loss: 
# 0.1631 

