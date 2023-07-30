import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and scaler
with open('finalized_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the HTML form
        input_data = [float(request.form[f'input{i}']) for i in range(1, 7)]

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform([input_data])

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_data)

        # You can process the prediction further if needed

        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        return render_template('index.html', error_message='Error: ' + str(e))


if __name__ == '__main__':
    app.run(debug=True)
