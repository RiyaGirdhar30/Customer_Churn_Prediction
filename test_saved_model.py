import joblib
import pandas as pd

# Load the saved model
model = joblib.load("churn_model.pkl")

# Load the dataset to get one example customer
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Prepare the customer's features
customer = df.drop(columns=["customerID", "Churn"]).iloc[[0]].copy()

# Convert TotalCharges to numeric
customer["TotalCharges"] = pd.to_numeric(
    customer["TotalCharges"],
    errors="coerce"
)

# Fill missing charges for a zero-tenure customer
customer.loc[
    (customer["tenure"] == 0) & (customer["TotalCharges"].isna()),
    "TotalCharges"
] = 0

# Predict churn
prediction = model.predict(customer)[0]

if prediction == 1:
    print("Prediction: Customer may churn.")
else:
    print("Prediction: Customer is likely to stay.")
