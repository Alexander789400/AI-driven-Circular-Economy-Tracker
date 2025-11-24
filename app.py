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

# Sector mapping
sectors = {
    "Automotive": 0,
    "Electronics": 1,
    "Manufacturing": 2,
    "FoodProcessing": 3,
    "Textiles": 4
}

cols_to_scale = [
    'production_volume_units',
    'raw_material_kg',
    'recycled_material_kg',
    'waste_kg',
    'energy_kwh',
    'water_liters'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_class = None
    prediction_regression = None
    details = None
    
    if request.method == 'POST':
        try:
            # Get feature inputs from user
            production_volume_units = float(request.form['production_volume_units'])
            raw_material_kg = float(request.form['raw_material_kg'])
            recycled_material_kg = float(request.form['recycled_material_kg'])
            waste_kg = float(request.form['waste_kg'])
            energy_kwh = float(request.form['energy_kwh'])
            water_liters = float(request.form['water_liters'])
            machine_downtime_hours = float(request.form['machine_downtime_hours'])
            sector = request.form['sector']

            # Calculate derived features
            material_recovery_rate = recycled_material_kg / (recycled_material_kg + waste_kg)  # avoid divide by zero
            machine_downtime_ratio = (machine_downtime_hours * 60) / 1440  # convert hours to ratio

            # Encode sector
            sector_encoded = sectors.get(sector, 0)

            # Prepare feature array
            features = [
                production_volume_units,
                raw_material_kg,
                recycled_material_kg,
                waste_kg,
                energy_kwh,
                water_liters,
                material_recovery_rate,
                sector_encoded,
                machine_downtime_ratio
            ]

            # Scale required features
            features_scaled = np.array(features).reshape(1, -1)
            features_scaled[:, :6] = scaler.transform(features_scaled[:, :6])  # scale only first 6 features

            # Predictions
            prediction_class_num = logistic_model.predict(features_scaled)[0]
            prediction_class = "High waste" if prediction_class_num == 1 else "Low waste"
            prediction_regression = ridge_model.predict(features_scaled)[0]

            details = {
                "Waste Level (Classification)": prediction_class,
                "Circularity Score (Regression)": round(prediction_regression, 2),
                "Material Recovery Rate": round(material_recovery_rate, 3),
                "Machine Downtime Ratio": round(machine_downtime_ratio, 3)
            }

        except Exception as e:
            details = {"Error": str(e)}

    return render_template('index.html', sectors=sectors.keys(),
                           details=details)

if __name__ == '__main__':
    app.run(debug=True)
