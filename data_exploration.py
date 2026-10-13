import pandas as pd

# Load the customer churn dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# Display the first 5 rows
print(df.head())

# Display the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display all column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types and non-null counts
print("\nDataset Information:")
df.info()

# Check missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check TotalCharges data type
print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

# Check for blank values in TotalCharges
print("\nBlank TotalCharges Values:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check the result
print("\nTotalCharges Data Type After Conversion:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges After Conversion:")
print(df["TotalCharges"].isnull().sum())

# Inspect customers with missing TotalCharges
missing_charges = df[df["TotalCharges"].isnull()]

print("\nCustomers With Missing TotalCharges:")
print(
    missing_charges[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\nTenure Values for Missing Charges:")
print(missing_charges["tenure"].value_counts())

# Fill missing TotalCharges for customers with zero tenure
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isnull()),
    "TotalCharges"
] = 0

# Verify the cleaning
print("\nMissing TotalCharges After Cleaning:")
print(df["TotalCharges"].isnull().sum())

print("\nTotalCharges for Zero-Tenure Customers:")
print(
    df.loc[
        df["tenure"] == 0,
        ["customerID", "tenure", "TotalCharges", "Churn"]
    ]
)


# Analyze customer churn distribution
print("\nCustomer Churn Distribution:")
print(df["Churn"].value_counts())

# Calculate churn percentages
print("\nCustomer Churn Percentage:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# Calculate the overall churn rate
churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")


import matplotlib.pyplot as plt

# Count customers in each churn category
churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])

# Create a bar chart
plt.figure(figsize=(7, 5))
churn_counts.plot(kind="bar", color=["seagreen", "tomato"])

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the chart
plt.savefig("churn_distribution.png", dpi=300, bbox_inches="tight")

# Display the chart
plt.show()


# Analyze churn by contract type
print("\nChurn Rate by Contract Type:")

contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda group: (group == "Yes").mean() * 100)
    .round(2)
)

print(contract_churn)


# Visualize churn rate by contract type
plt.figure(figsize=(8, 5))

contract_churn.plot(
    kind="bar",
    color=["tomato", "seagreen", "seagreen"]
)

plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.ylim(0, 50)
plt.tight_layout()

plt.savefig(
    "contract_churn_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Analyze customer tenure by churn status
print("\nAverage Tenure by Churn Status:")

tenure_by_churn = df.groupby("Churn")["tenure"].mean().round(2)

print(tenure_by_churn)


# Visualize average tenure by churn status
plt.figure(figsize=(7, 5))

tenure_by_churn.plot(
    kind="bar",
    color=["seagreen", "tomato"]
)

plt.title("Average Customer Tenure by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Average Tenure (Months)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "tenure_by_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Analyze monthly charges by churn status
print("\nAverage Monthly Charges by Churn Status:")

charges_by_churn = (
    df.groupby("Churn")["MonthlyCharges"]
    .mean()
    .round(2)
)

print(charges_by_churn)


# Visualize average monthly charges by churn status
plt.figure(figsize=(7, 5))

charges_by_churn.plot(
    kind="bar",
    color=["seagreen", "tomato"]
)

plt.title("Average Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Average Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "monthly_charges_by_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Analyze churn by internet service
print("\nChurn Rate by Internet Service:")

internet_churn = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda group: (group == "Yes").mean() * 100)
    .round(2)
)

print(internet_churn)


# Visualize churn rate by internet service
plt.figure(figsize=(8, 5))

internet_churn.plot(
    kind="bar",
    color=["seagreen", "tomato", "slateblue"]
)

plt.title("Customer Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.ylim(0, 50)
plt.tight_layout()

plt.savefig(
    "internet_service_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Analyze churn by payment method
print("\nChurn Rate by Payment Method:")

payment_churn = (
    df.groupby("PaymentMethod")["Churn"]
    .apply(lambda group: (group == "Yes").mean() * 100)
    .round(2)
)

print(payment_churn)


# Visualize churn rate by payment method
plt.figure(figsize=(10, 6))

payment_churn.sort_values(ascending=False).plot(
    kind="bar",
    color=["tomato", "orange", "steelblue", "seagreen"]
)

plt.title("Customer Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 50)
plt.tight_layout()

plt.savefig(
    "payment_method_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Analyze churn by senior-citizen status
print("\nChurn Rate by Senior-Citizen Status:")

senior_churn = (
    df.groupby("SeniorCitizen")["Churn"]
    .apply(lambda group: (group == "Yes").mean() * 100)
    .round(2)
)

senior_churn.index = senior_churn.index.map({
    0: "Non-Senior Citizens",
    1: "Senior Citizens"
})

print(senior_churn)


# Visualize churn rate by senior-citizen status
plt.figure(figsize=(8, 5))

senior_churn.plot(
    kind="bar",
    color=["seagreen", "tomato"]
)

plt.title("Customer Churn Rate by Senior-Citizen Status")
plt.xlabel("Customer Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.ylim(0, 50)
plt.tight_layout()

plt.savefig(
    "senior_citizen_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Analyze churn by partner status
print("\nChurn Rate by Partner Status:")

partner_churn = (
    df.groupby("Partner")["Churn"]
    .apply(lambda group: (group == "Yes").mean() * 100)
    .round(2)
)

print(partner_churn)


# Visualize churn rate by partner status
plt.figure(figsize=(7, 5))

partner_churn.plot(
    kind="bar",
    color=["tomato", "seagreen"]
)

plt.title("Customer Churn Rate by Partner Status")
plt.xlabel("Has Partner")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.ylim(0, 40)
plt.tight_layout()

plt.savefig(
    "partner_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
