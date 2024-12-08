Recruitment Prediction App - README
Overview
The Recruitment Prediction App is an interactive machine-learning application designed to predict whether a student is likely to be recruited based on their academic performance and personal details. It leverages a Random Forest Classifier trained on a real-world dataset to provide accurate predictions, helping recruiters, HR professionals, and students make informed decisions.

How the App Works
User Inputs:
Users input their academic performance metrics (SSC, HSC, Degree, E-Test, and MBA percentages) and gender using a user-friendly sidebar interface.
Prediction:
Based on the provided inputs, the app predicts recruitment status as either:
Placed: Likely to be recruited.
Not Placed: Unlikely to be recruited.
Dataset Details:
Users can explore the dataset structure and model details through a dedicated navigation option.
Key Features
Data-Driven Predictions:
Predict recruitment outcomes using six key features.
Real-Time Results:
Instant predictions with high accuracy.
User-Friendly Interface:
Simplified navigation with a sidebar for inputs and navigation.
Model Transparency:
Detailed information on the dataset and Random Forest model, including accuracy and methodology.
System Requirements
Python 3.8 or higher
Required Python libraries:
streamlit
pandas
numpy
scikit-learn
Installation and Setup
Clone this repository or download the project files.
Install the required Python libraries:
bash
Copy code
pip install -r requirements.txt
Place the dataset (df4.csv) in the project directory.
Run the Streamlit app:
bash
Copy code
streamlit run app.py
Dataset Details
The app uses a dataset containing the following features:

Gender: Student's gender (Male/Female).
SSC Percentage: Secondary School Certificate percentage.
HSC Percentage: Higher Secondary Certificate percentage.
Degree Percentage: Undergraduate degree percentage.
E-Test Percentage: Aptitude test percentage.
MBA Percentage: MBA program percentage.
Status: Recruitment status (Placed/Not Placed).
Model Information
Algorithm: Random Forest Classifier
Accuracy: Achieved {accuracy * 100:.2f}% on test data.
Training:
Split the dataset into training and testing sets (80/20).
Features: Gender, SSC Percentage, HSC Percentage, Degree Percentage, E-Test Percentage, MBA Percentage.
Navigation Options
Predict Recruitment: Enter input data and receive recruitment predictions.
Details of Dataset and Model:
View dataset structure and sample data.
Understand the Random Forest model used for predictions.
About the Author
Developed by Tamoor Abbas as part of the Ineuron Project. This project demonstrates the application of machine learning in solving real-world recruitment challenges.

License
This project is licensed under the MIT License. Feel free to use and modify it.

Contact
For any queries or feedback:

Email: Tamur110@gmail.com
LinkedIn: Tamoor Abbas
Start using the app today and predict recruitment outcomes with confidence! 🎓
