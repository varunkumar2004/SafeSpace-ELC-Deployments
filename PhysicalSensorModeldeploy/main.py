from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('final.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# @app.route('/')
# def home():
    # return 'Hello, World!'
    # @app.route('/predict', )
@app.route('/',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            snoring_range = request.form.get('snoring_range')
            respiration_rate = request.form.get('respiration_rate')
            body_temperature = request.form.get('body_temperature')
            blood_oxygen = request.form.get('blood_oxygen')
            sleep = request.form.get('sleep')
            heart_rate = request.form.get('heart_rate')
            # return "Helo"
            if None in [snoring_range, respiration_rate, body_temperature, blood_oxygen, sleep, heart_rate]:
                return jsonify({'error': 'One or more input fields are missing'}), 400

            snoring_range = float(snoring_range)
            respiration_rate = float(respiration_rate)
            body_temperature = float(body_temperature)
            blood_oxygen = float(blood_oxygen)
            sleep = float(sleep)
            heart_rate = float(heart_rate)

            # Create an input query with the correct number of features
            input_query = np.array([[snoring_range, respiration_rate, body_temperature, blood_oxygen, sleep, heart_rate]])

            # # Predict the result
            result = model.predict(input_query)

            return jsonify({'Result': str(result[0])})
        
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    
    return 'Please send a POST request with the required parameters.'

if __name__ == '__main__':
    app.run(debug=True)
