
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# App title
st.title("🔌 Electricity Consumption Predictor")
st.markdown("Estimate monthly electricity consumption based on connected load, city, and month.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Pyyyy.csv')
    df['Month'] = df['Month'].astype(str)
    return df

df = load_data()

# Features & Target
X = df[['Connected_Load_kW', 'City', 'Month']]
y = df['Monthly_Consumption']

# Preprocessing and pipeline
categorical_features = ['City', 'Month']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
st.subheader("Model Evaluation")
st.write(f"R² Score: **{r2_score(y_test, y_pred):.4f}**")
st.write(f"RMSE: **{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}**")

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Consumption')
plt.ylabel('Predicted Consumption')
plt.title('Actual vs Predicted Monthly Consumption')
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
st.pyplot(plt)

# User Input
st.subheader("📥 Enter Customer Details")
connected_load = st.number_input("Connected Load (kW):", min_value=0.0, step=0.1)
city = st.selectbox("City:", sorted(df['City'].unique()))
month = st.selectbox("Month:", sorted(df['Month'].unique()))

# Predict
if st.button("🔮 Predict Monthly Consumption"):
    input_df = pd.DataFrame({
        'Connected_Load_kW': [connected_load],
        'City': [city],
        'Month': [month]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Monthly Consumption: **{prediction:.2f} units**")
