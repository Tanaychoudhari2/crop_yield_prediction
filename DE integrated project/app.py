from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and preprocessor
model = pickle.load(open('dtr.pkl', 'rb'))
preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        # Extract features from the form data
        year = int(data['Year'])
        rainfall = float(data['average_rain_fall_mm_per_year'])
        pesticides = float(data['pesticides_tonnes'])
        temperature = float(data['avg_temp'])
        area = data['Area']
        item = data['Item']

        # Prepare the input data for the model
        # Ensure Area and Item are handled using the loaded preprocessor
        features = np.array([[year, rainfall, pesticides, temperature, area, item]], dtype=object)
        
        # Preprocess the input features using the loaded preprocessor
        transformed_features = preprocesser.transform(features)
        
        # Get the prediction from the model
        prediction = model.predict(transformed_features)[0]
        
        # Return the prediction back to the HTML page
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
