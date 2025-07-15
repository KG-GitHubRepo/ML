# app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Title
st.set_page_config(page_title="🏠 House Price Predictor", layout='wide')
st.title("🏠 House Price Prediction App")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('kc_house_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['year_sold'] = data['date'].dt.year
    data['month_sold'] = data['date'].dt.month
    return data

data = load_data()

# Sidebar: choose model
st.sidebar.header("⚙️ Model & Features")
model_choice = st.sidebar.selectbox("Choose model:", ['Linear Regression', 'XGBoost'])

# Sidebar: input sliders
st.sidebar.header("🔧 Input Features")
features = {
    'sqft_living': st.sidebar.slider("sqft_living", 500, 6000, 2000),
    'grade': st.sidebar.slider("grade", 1, 13, 7),
    'bathrooms': st.sidebar.slider("bathrooms", 0, 8, 2),
    'sqft_above': st.sidebar.slider("sqft_above", 500, 6000, 1500),
    'view': st.sidebar.slider("view", 0, 4, 0),
    'sqft_basement': st.sidebar.slider("sqft_basement", 0, 3000, 500),
    'lat': st.sidebar.number_input("lat", value=47.5112),
    'waterfront': st.sidebar.selectbox("waterfront", [0,1]),
    'yr_built': st.sidebar.slider("yr_built", 1900, 2015, 1970),
    'bedrooms': st.sidebar.slider("bedrooms", 1, 10, 3)
}

# Prepare data
selected_features = list(features.keys())
X = data[selected_features]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train model
if model_choice == 'Linear Regression':
    model = LinearRegression()
else:
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

model.fit(X_train, y_train)

# Predict
input_df = pd.DataFrame([features])
pred = model.predict(input_df)[0]

st.subheader("📈 Predicted House Price")
st.success(f"{pred:,.0f}")
