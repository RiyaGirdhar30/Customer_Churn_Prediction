# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave the company based on customer demographics, service usage, contract details, and billing information.

## Features

- Customer churn prediction using a trained machine learning model
- Interactive web interface built with Streamlit
- 19 customer input features
- Estimated churn probability with a visual progress bar
- Customer information summary
- Input validation for zero-tenure customers
- Reset button to restore default customer details
- Exploratory data analysis with visualizations
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models

The project explores Logistic Regression and Decision Tree classification.

The models are evaluated on a held-out test dataset. The Streamlit application uses the saved model in `churn_model.pkl`.

## Project Structure

```text
Customer_Churn_Prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app.py
├── data_exploration.py
├── model_training.py
├── test_saved_model.py
├── churn_model.pkl
├── README.md
└── requirements.txt
```

Additional chart images generated during exploratory data analysis may also be present in the project folder.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/RiyaGirdhar30/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the application

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Dataset

The project uses the IBM Telco Customer Churn dataset, containing customer demographics, subscribed services, contract information, billing details, and churn labels.

## Important Note

The application produces model-based predictions and estimated probabilities. These results are not guarantees of future customer behavior.

## Future Improvements

- Improve churn detection and evaluate probability calibration
- Add model comparison visualizations
- Deploy the application online
- Improve the interface and prediction explanations

## Author

Riya Girdhar