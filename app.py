import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.pikbest.com/wp/202405/blur-red-background-with-blurred-houses-3d-illustration-of-a-group_9852180.jpg!bw700");
             background-attachment: fixed;
             background-size: cover;
         }}
         .stApp > header {{
             background-color: transparent;
         }}
         .stApp {{
             color: white;
         }}
         .stButton > button {{
             color: black;
             background-color: #ffffff;
             border: 2px solid #4CAF50;
         }}
         .stTextInput > div > div > input {{
             color: black;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Add background image
add_bg_from_url()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Housingprice.csv")

data = load_data()

# Preprocess dataset
@st.cache_data
def preprocess_data(data):
    data = data.drop(['Address'], axis=1)
    bins = [0, np.percentile(data['Price'], 33), np.percentile(data['Price'], 66), np.max(data['Price'])]
    labels = ['Low', 'Medium', 'High']
    data['Price_Category'] = pd.cut(data['Price'], bins=bins, labels=labels)
    data = data.drop(['Price'], axis=1)

    # Convert categorical variables to numerical using Label Encoding
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']):
        data[column] = label_encoder.fit_transform(data[column])

    return data

data = preprocess_data(data)

# Split data and train model
@st.cache_resource
def train_model(data):
    X = data.drop(['Price_Category'], axis=1)
    Y = data['Price_Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, Y_train)
    return rf_model

model = train_model(data)

# Define meaningful price ranges
price_ranges = {
    'Low': 'Affordable houses (below average prices)',
    'Medium': 'Moderate-priced houses',
    'High': 'Luxury or high-end houses'
}

# Streamlit app
st.title("House Price Prediction")

st.header("Input House Details")

# Using columns to place the input field and example side by side
col1, col2 = st.columns([3, 1])
with col1:
    income = st.number_input("Avg. Area Income:", min_value=0.0, step=1000.0, format="%.2f")
with col2:
    st.text("(e.g., 60000)")

col1, col2 = st.columns([3, 1])
with col1:
    age = st.number_input("Avg. Area House Age:", min_value=0.0, step=1.0, format="%.1f")
with col2:
    st.text("(e.g., 20)")

col1, col2 = st.columns([3, 1])
with col1:
    rooms = st.number_input("Avg. Area Number of Rooms:", min_value=0.0, step=1.0, format="%.1f")
with col2:
    st.text("(e.g., 5)")

col1, col2 = st.columns([3, 1])
with col1:
    bedrooms = st.number_input("Avg. Area Number of Bedrooms:", min_value=0.0, step=1.0, format="%.1f")
with col2:
    st.text("(e.g., 3)")

col1, col2 = st.columns([3, 1])
with col1:
    population = st.number_input("Area Population:", min_value=0.0, step=1000.0, format="%.0f")
with col2:
    st.text("(e.g., 36500)")

if st.button("Predict"):
    input_data = [[income, age, rooms, bedrooms, population]]
    prediction = model.predict(input_data)[0]
    predicted_category = price_ranges.get(prediction, 'Unknown')
    st.success(f'Predicted Price Category: {predicted_category}')

# Add some information about the model
st.sidebar.header("About")
st.sidebar.info("This app uses a Random Forest model to predict house price categories based on area statistics.")
st.sidebar.info("The model is trained on historical housing data and categorizes prices into Low, Medium, and High ranges.")
