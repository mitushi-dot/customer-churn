import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# Define feature preprocessing function
def preprocess_input(data):
    # Encode categorical features
    gender_map = {'Female': 0, 'Male': 1}
    yes_no_map = {'Yes': 1, 'No': 0}
    internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_method_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }

    # Apply mappings
    data['gender'] = gender_map[data['gender']]
    data['Partner'] = yes_no_map[data['Partner']]
    data['Dependents'] = yes_no_map[data['Dependents']]
    data['PhoneService'] = yes_no_map[data['PhoneService']]
    data['MultipleLines'] = yes_no_map[data['MultipleLines']]
    data['InternetService'] = internet_service_map[data['InternetService']]
    data['OnlineSecurity'] = yes_no_map[data['OnlineSecurity']]
    data['DeviceProtection'] = yes_no_map[data['DeviceProtection']]
    data['TechSupport'] = yes_no_map[data['TechSupport']]
    data['StreamingTV'] = yes_no_map[data['StreamingTV']]
    data['StreamingMovies'] = yes_no_map[data['StreamingMovies']]
    data['Contract'] = contract_map[data['Contract']]
    data['PaperlessBilling'] = yes_no_map[data['PaperlessBilling']]
    data['PaymentMethod'] = payment_method_map[data['PaymentMethod']]
    
    # Ensure numerical features are floats
    data['MonthlyCharges'] = float(data['MonthlyCharges'])
    data['TotalCharges'] = float(data['TotalCharges'])
    data['tenure'] = int(data['tenure'])

    # Convert to DataFrame for prediction
    return pd.DataFrame([data])

# Main Streamlit app
def main():
    st.title("Customer Churn Prediction")
    st.write("Fill out the customer details to predict whether they will churn.")

    # Collect user input
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (in months)", min_value=0, step=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    # Predict button
    if st.button("Predict"):
        # Preprocess input data
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # Preprocess and predict
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)

        # Display the result
        result = "Churn" if prediction[0] == 1 else "No Churn"
        st.success(f"The predicted result is: {result}")

if __name__ == "__main__":
    main()
