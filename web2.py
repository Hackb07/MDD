import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# Define PyTorch model class
class MoistureClassifier(nn.Module):
    def __init__(self):
        super(MoistureClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Load the trained model
model = MoistureClassifier()
model.load_state_dict(torch.load(r"/home/tharun/megs/python-app/moisture_model.pth"))
model.eval()

# Load the scaler
scaler = joblib.load(r"/home/tharun/megs/python-app/model.pkl")

# Streamlit UI Styling (Water-Themed)
st.set_page_config(page_title="Moisture Predictor", page_icon="💦", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;
            font-family: Arial, sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #0277bd;
        }
        .stButton>button {
            background-color: #0288d1 !important;
            color: white !important;
            font-size: 18px;
            padding: 12px;
            border-radius: 8px;
            border: none;
        }
        .stTextInput>div>div>input {
            border: 2px solid #0288d1 !important;
            border-radius: 5px;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #b3e5fc;
            color: #01579b;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 class='main-title'>💦 Moisture Collection Capability Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the weather parameters below to check if moisture collection is possible.")

# Input fields with water theme styling
col1, col2 = st.columns(2)
with col1:
    maxtempC = st.number_input("🌡 Maximum Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
    mintempC = st.number_input("❄️ Minimum Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

if st.button("🌊 Predict Moisture Capability"):
    # Convert input into a NumPy array
    input_data = np.array([[maxtempC, mintempC, humidity]], dtype=np.float32)
    
    # Normalize input data
    input_data_scaled = scaler.transform(input_data)
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Interpret result
    result = "💧 Capable" if prediction > 0.5 else "⚠️ Not Capable"
    
    # Display result with a water-themed box
    st.markdown(f"<div class='result-box'>Moisture Collection Capability: {result}</div>", unsafe_allow_html=True)