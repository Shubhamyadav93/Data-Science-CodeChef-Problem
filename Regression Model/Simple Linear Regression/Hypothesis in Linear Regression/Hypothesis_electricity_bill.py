import pandas as pd 
from sklearn.linear_model import LinearRegression

class ElectricityBillPrediction:
    """
        Train a Simple Linear Regression 
        model and use the learned 
        hypothesis to predict the electricity bill
    """


    def __init__(self):
        self.dataset = pd.read_csv(
            '/Users/shubham/Desktop/Fundamentals of Data Science/CodeChefProblem/Regression Model/Simple Linear Regression/Hypothesis in Linear Regression/electricity_bill.csv'
        )

    def prepare_data(self):

        # Store the input feature 
        self.X = self.dataset[["UnitsConsumed"]]
        self.y = self.dataset["ElectricityBill"]

    def train_model(self):

        # Create the Linear Regression model 
        self.model = LinearRegression()
        # Train the model 
        self.model.fit(self.X,self.y)

    def get_hypothesis(self):

        # Retrive the intercept 
        intercept = self.model.intercept_

        # Retrive the slope 
        slope = self.model.coef_[0]

        return{
            "intercept":intercept,
            "slope":slope
        }

    def predict_bill(self):

        # Prediction the electricity bill
        input_df = pd.DataFrame({"UnitsConsumed":[350]})
        prediction = self.model.predict(input_df)
        return prediction[0]


if __name__ == "__main__":

    predictor = ElectricityBillPrediction()

    predictor.prepare_data()

    predictor.train_model() 

    hypothesis = predictor.get_hypothesis()

    print("Intercept (b0:)",hypothesis["intercept"])
    print("Slop (b1):",hypothesis["slope"])

    print("\nHypothesis:")
    print(
        f"ŷ = {hypothesis['intercept']} + " 
        f"({hypothesis['slope']} * x )"
    )


    print("\nUnits Consumed: 350")
    print("Predicted Electricity Bill:",predictor.predict_bill())




# Left side (CodeChef platform) par output aisa dikh raha hai:
"""
Left side (CodeChef platform) par output aisa dikh raha hai:
Intercept ($b_0$): 4.547473508864641e-13
Slope ($b_1$): 7.999999999999999
Jabki right side (aapke VS Code) par output bilkul clean round-off hokar aa raha hai:
Intercept ($b_0$): 0.0
Slope ($b_1$): 8.0
Iska karan yeh hai ki dono jagah Python aur Scikit-Learn ke versions alag-alag ho sakte hain.
CodeChef ke cloud server par jo library ka version chal raha hai, wahan floating-point calculation karte waqt computer ki internal binary precision ki wajah se thoda sa rounding error aa jata hai (jaise 7.999999999999999 jo ki asliyat mein 8.0 hi hai).
Jabki aapke Mac par jo Python/scikit-learn ka version instal hai, wo us value ko seedhe clean format mein ya slightly different internal precision ke sath evaluate kar raha hai.
Dono outputs ka mathematical matlab 100% ek hi hai. Machine learning mein itna chhota sa difference (e-13 matlab zero ke barabar) bilkul normal hai aur isse aapke code par ya prediction (2800.0) par koi asar nahi padega. Aapka code bilkul sahi chal raha hai!

"""



