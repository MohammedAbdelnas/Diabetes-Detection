from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
from flask_restful import Resource , Api

app = Flask(__name__)

CORS(app)
api = Api(app)
# Load your model
model = joblib.load(open("Final Projects\Diabetes ML\models\Diabetes.pK1",'rb'))

@app.route('/')
def home():
    return "Diabetes Detection API"

@app.route('/predict', methods=['POST'])
def predict():
        # Extract features from request
        data = request.json
        features = np.array([[
            data['pregnancies'],
            data['glucose'],
            data['blood_pressure'],
            data['skin_thickness'],
            data['insulin'],
            data['bmi'],
            data['diabetes_pedigree'],
            data['age']
        ]])
        
        # Predict using the loaded model
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
