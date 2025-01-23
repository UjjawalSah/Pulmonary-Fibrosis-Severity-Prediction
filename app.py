import flask
from flask import Flask, request, jsonify, render_template
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

# Flask app setup
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

# Load the model (ensure this is compatible for serverless use)
model = joblib.load("fvc_transformer_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        fvc_values = [
            float(data['fvc1']),
            float(data['fvc2']),
            float(data['fvc3']),
            float(data['fvc4']),
            float(data['fvc5'])
        ]

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(fvc_values, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)

        # Convert prediction to list
        prediction_list = prediction.squeeze(0).tolist()

        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# No need for app.run() in serverless environments
