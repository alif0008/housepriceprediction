import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("Housingprice.csv")

# Preprocess dataset
data.drop(['Address'], axis=1, inplace=True)
bins = [0, np.percentile(data['Price'], 33), np.percentile(data['Price'], 66), np.max(data['Price'])]
labels = ['Low', 'Medium', 'High']
data['Price_Category'] = pd.cut(data['Price'], bins=bins, labels=labels)
data.drop(['Price'], axis=1, inplace=True)

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])

# Split data into train and test sets
X = data.drop(['Price_Category'], axis=1)
Y = data['Price_Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Train RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)

# Define meaningful price ranges
price_ranges = {
    'Low': 'Affordable houses (below average prices)',
    'Medium': 'Moderate-priced houses',
    'High': 'Luxury or high-end houses'
}

# Streamlit app
st.title("House Price Prediction")

st.header("House Price Prediction")

income = st.number_input("Avg. Area Income:", min_value=0.0, step=1.0)
age = st.number_input("Avg. Area House Age:", min_value=0.0, step=1.0)
rooms = st.number_input("Avg. Area Number of Rooms:", min_value=0.0, step=1.0)
bedrooms = st.number_input("Avg. Area Number of Bedrooms:", min_value=0.0, step=1.0)
population = st.number_input("Area Population:", min_value=0.0, step=1.0)

if st.button("Predict"):
    input_data = [[income, age, rooms, bedrooms, population]]
    prediction = rf_model.predict(input_data)[0]
    predicted_category = price_ranges.get(prediction, 'Unknown')
    st.success(f'Predicted Price Category: {predicted_category}')
