from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'G_9_oil_category_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        feature_names = [
            'Day', 'DayOfWeek', 'month', 'year', 'IsWeekend', 
            'IsStartOfMonth', 'prev_1', 'prev_3', 'prev_7', 
            'avr_3', 'avr_7', 'Sales_Growth_Rate'
        ]
        form_data = {feature: float(request.form[feature]) for feature in feature_names}
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame([form_data])
        
        # Predict using the model
        prediction = model.predict(input_data)
        output = f"{prediction[0]:,.2f}"  # Format the result
        
        return render_template('index.html', prediction_text=f"Predicted Sales: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
