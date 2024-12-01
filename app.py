from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load('phishing_detection_ensemble_model.sav')  # Make sure this file exists in your project folder

# Feature extraction function
def extract_features(url):
    length = len(url)
    num_dots = url.count('.')
    has_https = int('https' in url)
    num_subdomains = url.count('.') - 1
    return [length, num_dots, has_https, num_subdomains]

# Define the home route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')  # You'll need to create this file

# Define a route to handle form submission and show results
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = request.form['url']  # Change 'url' to the name of your input field in index.html
        
        # Process the input (extract features)
        features = extract_features(user_input)
        
        # Make prediction
        prediction = model.predict([features])
        
        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction[0])  # Create result.html for displaying the result
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
