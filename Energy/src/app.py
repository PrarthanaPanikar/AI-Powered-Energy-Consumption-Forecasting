from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')   # make sure model.pkl is in same folder

# API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get input values
    hour = data['hour']
    day = data['day']

    # Convert to array
    features = np.array([[hour, day]])

    # Predict
    prediction = model.predict(features)

    # Return result
    return jsonify({'predicted_energy': float(prediction[0])})

# Run server
if __name__ == '__main__':
    app.run(debug=True)