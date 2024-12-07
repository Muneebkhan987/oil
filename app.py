from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'model1.pkl'  # Update to your actual model file name
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    sales_growth_rate = float(request.form['sales_growth_rate'])
    prev_3_sale = float(request.form['Prev_3_sale'])

    # Prepare the features as required by your model
    features = np.array([[sales_growth_rate, prev_3_sale]])

    # Make prediction
    prediction = model.predict(features)
    output = f'Predicted Sales: {prediction[0]}'

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
