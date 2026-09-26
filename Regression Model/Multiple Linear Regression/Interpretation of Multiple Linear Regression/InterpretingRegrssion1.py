import pandas as pd 
from sklearn.linear_model import LinearRegression

def load_dataset():
    """
        Load the employee salary dataset.

    """

    dataset = pd.read_csv('/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Multiple Linear Regression/Interpretation of Multiple Linear Regression/employee_salary1.csv')

    return dataset

def prepare_dat(dataset):
    """
    Separate the input features and target value.
    
    """
    X = dataset[["Experience","Certifications","EducationLevel"]]

    y = dataset["Salary"]

    return X,y 

def train_model(X,y):
    """
    Train the Multiple Linear Regression model.
    """
    model = LinearRegression()

    model.fit(X,y) 

    return model

def display_regression_results(model,feature_names):
    """
        Display the regression equation,
        intercept,coefficients,and feature effects.
    """

    intercept = model.intercept_

    coefficients = model.coef_

    print("Intercept:")
    print(f"{intercept:.2f}")

    print("\n Regression Equation:")
    equation = (
        f"Salary = {intercept:.2f}"
        f" + ({coefficients[0]:.2f} * Experience)"
        f" + ({coefficients[1]:.2f} * Certifications)"
        f" + ({coefficients[2]:.2f}) * EducationLevel"
    )

    print(equation)

    print("\nFeature Effects:")

    for feature,coefficient in zip(feature_names,coefficients):
        effect = "Positive" if coefficient >= 0 else "Negative"
        print(
            f"{feature}: "
            f"{coefficient:.2f}"
            f"({effect})"
        )

def main():
    dataset = load_dataset()

    X,y = prepare_dat(dataset)

    model = train_model(X,y)

    feature_name = [
        "Experience",
        "Certifications",
        "EducationLevel"
    ]

    display_regression_results(model,feature_name)

if __name__ == "__main__":
    main()


