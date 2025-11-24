from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models and preprocessors
with open('best_logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('best_ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_sector.pkl', 'rb') as f:
    le_sector = pickle.load(f)

# Sector mapping for user display
sectors = {
    "Automotive": 0,
    "Electronics": 1,
    "Manufacturing": 2,
    "FoodProcessing": 3,
    "Textiles": 4
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_class = None
    prediction_regression = None
    details = None
    
    if request.method == 'POST':
        # Get form data
        feature_values = []
        for i in range(1, 10):
            value = request.form.get(f'feature{i}')
            try:
                feature_values.append(float(value))
            except:
                feature_values.append(0.0)
        
        sector = request.form.get('sector')
        sector_encoded = sectors.get(sector, 0)
        feature_values.append(sector_encoded)

        # Convert to numpy array and scale
        features_array = np.array(feature_values).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Predictions
        prediction_class_num = logistic_model.predict(features_scaled)[0]
        prediction_class = le_sector.inverse_transform([prediction_class_num])[0]
        prediction_regression = ridge_model.predict(features_scaled)[0]

        details = {
            "Classification (Sector)": prediction_class,
            "Regression (Predicted Value)": round(prediction_regression, 2)
        }

    return render_template('index.html', sectors=sectors.keys(),
                           prediction_class=prediction_class,
                           prediction_regression=prediction_regression,
                           details=details)

if __name__ == '__main__':
    app.run(debug=True)
