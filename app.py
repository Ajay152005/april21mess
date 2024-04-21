from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

# Define route for home page

@app.route('/')

def home():
    return render_template('index1.html')

# Define route for prediction

@app.route('/predict', methods=['POST'])

def predict():
    # Get input from the form
    features = [float(x) for x in request.form.values()]

    # Preprocess the input data
    input_data = np.array(features).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Map numeric prediction to class name

    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species[prediction]

    # Render result template with prediction

    return render_template('result.html', prediction=predicted_species)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)