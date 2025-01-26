import streamlit as st
import joblib
import torch
import torch.nn as nn

# Define the FVCTransformer class
class FVCTransformer(nn.Module):
    def __init__(self, fvc_input_dim, output_dim):
        super(FVCTransformer, self).__init__()
        self.fvc_linear = nn.Linear(fvc_input_dim, 64)
        self.output_linear = nn.Linear(64, output_dim)

    def forward(self, fvc):
        fvc_output = self.fvc_linear(fvc)
        output = self.output_linear(fvc_output)
        return output

# Load the trained model
model = joblib.load("fvc_transformer_model.pkl")

# Streamlit app setup
st.set_page_config(page_title="Pulmonary Fibrosis Prediction", layout="centered")

# Page Title and Description
st.markdown(
    """
    <h1 style='text-align: center; color: #007BFF;'>Pulmonary Fibrosis Prediction</h1>
    <p style='text-align: center; color: #555;'>This application uses advanced machine learning techniques to predict Forced Vital Capacity (FVC) values for patients with pulmonary fibrosis. 
    With easy-to-use input fields, it helps monitor the progression of pulmonary fibrosis, offering valuable insights to healthcare professionals.</p>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.subheader("Enter FVC Values")
with st.form("input_form"):
    fvc1 = st.text_input("FVC Value 1:")
    fvc2 = st.text_input("FVC Value 2:")
    fvc3 = st.text_input("FVC Value 3:")
    fvc4 = st.text_input("FVC Value 4:")
    fvc5 = st.text_input("FVC Value 5:")
    submitted = st.form_submit_button("Predict")

# Handle prediction on form submission
if submitted:
    try:
        # Prepare input tensor
        fvc_values = [float(fvc1), float(fvc2), float(fvc3), float(fvc4), float(fvc5)]
        input_tensor = torch.tensor(fvc_values, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)

        # Convert prediction to a readable format
        prediction_list = prediction.squeeze(0).tolist()

        # Display the prediction
        st.success("Prediction Successful!")
        st.write("Predicted Values:")
        st.write(prediction_list)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# How It Works Section
st.markdown("---")
st.subheader("How It Works")
st.markdown(
    """
    This prediction model analyzes FVC values from the past and predicts future values. It has been trained using a large dataset 
    (over **20GB**) that includes both tabular data and CT scan images. This combination enables the model to understand the 
    relationship between past FVC values and visual lung data, leading to highly accurate predictions.

    The model is designed to be easy to use, even for those with no technical background. Simply input the past FVC values, and 
    the system will provide the predicted values, which are crucial for healthcare decision-making.
    """
)

# Key Features Section
st.markdown("### Key Features")
st.markdown(
    """
    - Predicts FVC values based on **5 input measurements**.
    - Trained on a **large 20GB dataset** combining tabular and CT scan data.
    - Outputs **accurate predictions** based on your input values.
    - Designed to be **user-friendly and easy to understand** for non-technical users.
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: #555;'>For more details and to view the code, visit the project repository on 
    <a href="https://github.com/UjjawalSah/Pulmonary-Fibrosis-Severity-Prediction" style="color: #007BFF;" target="_blank">GitHub</a>.</p>
    """,
    unsafe_allow_html=True,
)
