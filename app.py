import streamlit as st
import joblib
import pandas as pd

# Reset all customer input fields

def reset_customer_details():
    defaults = {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.0,
        "total_charges": 840.0,
    }

    for key, value in defaults.items():
        st.session_state[key] = value

# Load the trained machine learning model
model = joblib.load("churn_model.pkl")

# Configure the page
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.button(
    "Reset Customer Details",
    on_click=reset_customer_details
)

# App title and description
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "the company based on their details."
)

st.divider()

st.subheader("Enter Customer Details")

# Customer Profile
st.markdown("### 👤 Customer Profile")

profile_col1, profile_col2 = st.columns(2)

with profile_col1:
    if "gender" not in st.session_state:
        st.session_state["gender"] = "Female"

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        key="gender"
    )

    if "senior_citizen" not in st.session_state:
         st.session_state["senior_citizen"] = "No"
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
        key="senior_citizen"
    )

    if "partner" not in st.session_state:
        st.session_state["partner"] = "Yes"
    partner = st.selectbox(
        "Has Partner?",
        ["Yes", "No"],
        key="partner"
    )

with profile_col2:

    if "dependents" not in st.session_state:
        st.session_state["dependents"] = "No"
    dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"],
        key="dependents"
    )

    if "tenure" not in st.session_state:
        st.session_state["tenure"] = 12
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        key="tenure"
    )

# Services section
st.markdown("### 📡 Services")

service_col1, service_col2 = st.columns(2)

with service_col1:

    if "phone_service" not in st.session_state:
        st.session_state["phone_service"] = "Yes"
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
        key="phone_service"
    )

    if "multiple_lines" not in st.session_state:
        st.session_state["multiple_lines"] = "No"
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"],
        key="multiple_lines"
    )

    if "internet_service" not in st.session_state:
        st.session_state["internet_service"] = "DSL"
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        key="internet_service"
    )

    if "online_security" not in st.session_state:
        st.session_state["online_security"] = "No"
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"],
        key="online_security"
    )

    if "online_backup" not in st.session_state:
        st.session_state["online_backup"] = "No"
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"],
        key="online_backup"
    )

with service_col2:

    if "device_protection" not in st.session_state:
        st.session_state["device_protection"] = "No"
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"],
        key="device_protection"
    )

    if "tech_support" not in st.session_state:
        st.session_state["tech_support"] = "No"
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"],
        key="tech_support"
    )

    if "streaming_tv" not in st.session_state:
        st.session_state["streaming_tv"] = "No"
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"],
        key="streaming_tv"
    )

    if "streaming_movies" not in st.session_state:
        st.session_state["streaming_movies"] = "No"
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
        key="streaming_movies"
    )

# Billing and Contract section
st.markdown("### 💳 Billing & Contract")

billing_col1, billing_col2 = st.columns(2)

with billing_col1:

    if "contract" not in st.session_state:
        st.session_state["contract"] = "Month-to-Month"
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        key="contract"
    )

    if "paperless_billing" not in st.session_state:
        st.session_state["paperless_billing"] = "Yes"
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
        key="paperless_billing"
    )

    if "payment_method" not in st.session_state:
        st.session_state["payment_method"] = "Electronic check"
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        key="payment_method"
    )

with billing_col2:

    if "monthly_charges" not in st.session_state:
        st.session_state["monthly_charges"] = 70.0
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        key="monthly_charges"
    )

    if "total_charges" not in st.session_state:
        st.session_state["total_charges"] = 840.0
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        step=10.0,
        key="total_charges"
    )

# Prepare customer data for prediction
customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

st.subheader("Prepared Customer Data")
st.write(customer_data)

# Predict customer churn
st.divider()

if st.button("Predict Churn", type="primary"):
    if tenure == 0 and total_charges > 0:
        st.error(
            "Please set Total Charges to 0 when tenure is 0."
        )
    else:
        prediction = model.predict(customer_data)[0]

        probabilities = model.predict_proba(customer_data)[0]
        churn_probability = (
            probabilities[list(model.classes_).index(1)] * 100
        )

        st.subheader("Prediction Result")
    
        st.caption(
            "This result is generated by a machine learning model "
            "trained on historical telecom customer data."
        )

        if prediction == 1:
            st.info(
                "Suggested action: Review the customer's contract, "
                "service experience, and support needs."
            )
        else:
            st.info(
                "Suggested action: Continue monitoring customer "
                "satisfaction and service usage."
            )


        if prediction == 1:
            st.error("⚠️ This customer is predicted to churn.")
        else:
            st.success("✅ This customer is predicted to stay.")

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.2f}%"
        )

        st.progress(int(round(churn_probability)))

# Validate customer details
if tenure == 0 and total_charges > 0:
    st.warning(
        "A customer with zero months of tenure should have "
        "zero total charges. Please review the values."
    )

# Display the entered information
st.divider()
st.subheader("Customer Information Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.write("**Gender:**", gender)
    st.write("**Senior Citizen:**", senior_citizen)
    st.write("**Partner:**", partner)
    st.write("**Dependents:**", dependents)
    st.write("**Tenure:**", tenure, "months")
    st.write("**Phone Service:**", phone_service)
    st.write("**Multiple Lines:**", multiple_lines)
    st.write("**Internet Service:**", internet_service)
    st.write("**Contract:**", contract)

with summary_col2:
    st.write("**Online Security:**", online_security)
    st.write("**Online Backup:**", online_backup)
    st.write("**Device Protection:**", device_protection)
    st.write("**Tech Support:**", tech_support)
    st.write("**Streaming TV:**", streaming_tv)
    st.write("**Streaming Movies:**", streaming_movies)
    st.write("**Paperless Billing:**", paperless_billing)
    st.write("**Payment Method:**", payment_method)
    st.write("**Monthly Charges:** $", monthly_charges)
    st.write("**Total Charges:** $", total_charges)

# Footer
st.divider()

st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Customer Churn Prediction | Built with Python, "
    "Pandas, Scikit-learn & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
