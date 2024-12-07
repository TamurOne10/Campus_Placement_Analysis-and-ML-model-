import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df4 = pd.read_csv('df4.csv')

# Feature selection
selected_features = ['gender', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
target_column = 'status'

# Prepare data
X = df4[selected_features]
y = df4[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("🎓 Recruitment Prediction App")
st.markdown("🌟 Predict recruitment chances based on academic and personal details!")
st.markdown("🌟 Created by **Tamoor Abbas** using a Random Forest Classifier for Ineuron Project.")
st.markdown("### 💡 **How to Use the App**")
st.markdown("🔍 **Enter your details in the sidebar**, including academic percentages and gender. Then click **'Predict Recruitment'** to see your result!")

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Recruitment", "Details of Dataset and Model"])

# Show "Predict Recruitment" page by default
if page == "Predict Recruitment":
    # Sidebar inputs for prediction
    st.sidebar.header("Input Academic Percentages and Gender")
    gender = st.sidebar.selectbox("Select Gender", ['Male', 'Female'], index=0)
    ssc_p = st.sidebar.number_input("Enter Secondary School Certificate (SSC) Percentage (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    hsc_p = st.sidebar.number_input("Enter Higher Secondary Certificate (HSC) Percentage (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    degree_p = st.sidebar.number_input("Enter Degree Percentage (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    etest_p = st.sidebar.number_input("Enter E-Test Percentage (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    mba_p = st.sidebar.number_input("Enter MBA Percentage (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    if st.sidebar.button("Predict Recruitment"):
        # Convert gender to numeric
        gender_numeric = 1 if gender == 'Male' else 0
        input_data = np.array([[gender_numeric, ssc_p, hsc_p, degree_p, etest_p, mba_p]])

        # Check for NaN or invalid inputs
        if np.isnan(input_data).any():
            st.error("❌ **Error: All input fields must be filled correctly!**")
        else:
            try:
                prediction = rf_model.predict(input_data)[0]
                st.subheader("Recruitment Prediction Result")
                if prediction == "Placed":
                    st.success("✅ **Congratulations! The Candidate is likely to be Recruited!**")
                elif prediction == "Not Placed":
                    st.error("❌ **Unfortunately, the Candidate is not likely to be Recruited.**")
            except ValueError as e:
                st.error(f"❌ **An error occurred during prediction: {e}**")

# Show "Details of Dataset and Model" page
if page == "Details of Dataset and Model":
    st.header("📊 Dataset and Model Details")
    
    # Display Dataset Info
    st.subheader("Dataset Information")
    st.markdown("The dataset contains the following columns:")
    st.markdown("1. **Gender**: The gender of the student (Male/Female)")
    st.markdown("2. **SSC Percentage**: The percentage of marks scored in the Secondary School Certificate.")
    st.markdown("3. **HSC Percentage**: The percentage of marks scored in the Higher Secondary Certificate.")
    st.markdown("4. **Degree Percentage**: The percentage of marks scored in the undergraduate degree program.")
    st.markdown("5. **E-Test Percentage**: The percentage scored in the E-test.")
    st.markdown("6. **MBA Percentage**: The percentage of marks scored in an MBA program.")
    st.markdown("7. **Status**: The recruitment status of the student (Placed/Not Placed).")

    # Display Dataset sample
    st.subheader("Sample of the Dataset")
    st.dataframe(df4.head())

    # Display Model Info
    st.subheader("Random Forest Model")
    st.markdown("The model used is a **Random Forest Classifier**, which is an ensemble learning method for classification.")
    st.markdown("It works by creating multiple decision trees and aggregating their results.")
    st.markdown("The model was trained with the following features: Gender, SSC Percentage, HSC Percentage, Degree Percentage, E-Test Percentage, and MBA Percentage.")
    st.markdown(f"**Model Accuracy**: {accuracy * 100:.2f}%")

# Adding Details about features of App 
st.title("Detail of features")
st.markdown("""
Welcome to the **Recruitment Prediction App** – a powerful tool designed to predict whether a student will be recruited based on their academic performance and personal data. This app utilizes machine learning techniques, specifically a **Random Forest Classifier**, to provide insights into recruitment trends.

**How does it work?**
- The app takes into account key academic metrics such as SSC, HSC, Degree, E-test, and MBA percentages, along with gender, to predict whether a candidate is likely to be recruited or not.
- The model has been trained using real-world data to accurately predict recruitment chances.

**Why this App?**
- This tool is specifically designed to assist recruiters, HR professionals, and students in making data-driven decisions based on academic performance.
- The Random Forest Classifier has been selected for its robustness and ability to handle various types of data, ensuring high accuracy and reliability in predictions.

### Key Features:
- **Data-Driven Predictions**: Predict the recruitment status (Placed/Not Placed) based on multiple input parameters.
- **User-Friendly Interface**: Easily input academic percentages and other relevant details.
- **Real-Time Results**: Receive instant predictions based on the latest model accuracy.

**Accuracy of the Model:**
The Random Forest model has been trained and evaluated on historical data, achieving an accuracy of **{accuracy * 100:.2f}%**, demonstrating its effectiveness in predicting recruitment outcomes.

This app is built by **Tamoor Abbas** for the **Ineuron Project** and is a perfect demonstration of how AI can be applied to real-world recruitment challenges. Whether you're a recruiter looking to optimize hiring decisions or a student aiming to understand your recruitment chances better, this app is here to provide valuable insights.

Start exploring and see if you're eligible for recruitment based on your academic profile!
""")
