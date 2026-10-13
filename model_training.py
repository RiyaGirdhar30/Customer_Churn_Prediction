import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Convert TotalCharges to a numerical column
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing charges for customers with zero tenure
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isna()),
    "TotalCharges"
] = 0

# Display the dataset shape
print("Dataset shape:", df.shape)

# Display the first five rows
print("\nFirst five rows:")
print(df.head())

# Display the target column distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())


# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# Display the shapes
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Display feature column names
print("\nFeature columns:")
print(X.columns.tolist())

# Display target values
print("\nTarget values:")
print(y.unique())


# Convert target labels into numbers
y = y.map({"No": 0, "Yes": 1})

# Display the converted target values
print("\nTarget values after conversion:")
print(y.unique())

# Check the target data type
print("\nTarget data type:", y.dtype)


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the shapes of the split datasets
print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of numerical columns:", len(numerical_columns))
print("Number of categorical columns:", len(categorical_columns))

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

print("\nPreprocessing pipeline created successfully!")

# Create the complete machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Display detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create a Decision Tree model
tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

# Train the Decision Tree model
tree_model.fit(X_train, y_train)

# Make predictions
tree_pred = tree_model.predict(X_test)

# Evaluate the Decision Tree
print("\n--- Decision Tree Results ---")
print("Accuracy:", round(accuracy_score(y_test, tree_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))

# Save the trained Logistic Regression pipeline
joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully as churn_model.pkl!")