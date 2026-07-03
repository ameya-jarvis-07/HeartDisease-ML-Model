from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

try:
    model = joblib.load('best_heart_disease_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        prediction = model_pipeline.predict(input_df)
        result = int(prediction[0])
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)