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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_class = None
    prediction_regression = None
    details = None
    recommendations = None

    if request.method == 'POST':
        try:
            # Inputs
            production = float(request.form['production_volume_units'])
            raw = float(request.form['raw_material_kg'])
            rec = float(request.form['recycled_material_kg'])
            waste = float(request.form['waste_kg'])
            energy = float(request.form['energy_kwh'])
            water = float(request.form['water_liters'])
            downtime_hours = float(request.form['machine_downtime_hours'])
            sector_text = request.form['sector']

            # Derived features
            material_rate = rec / (rec + waste) if (rec + waste) != 0 else 0
            downtime_ratio = (downtime_hours * 60) / 1440
            sector_encoded = sectors.get(sector_text, 0)

            # Scale only 6 features
            six_features = np.array([[production, raw, rec, waste, energy, water]])
            six_scaled = scaler.transform(six_features)

            # Final feature vector (correct order!)
            final_features = np.hstack([
                six_scaled,
                np.array([[material_rate, sector_encoded, downtime_ratio]])
            ])

            # Predict
            pred_class_num = logistic_model.predict(final_features)[0]
            prediction_class = "High Waste" if pred_class_num == 1 else "Low Waste"

            pred_reg = ridge_model.predict(final_features)[0]

            # Prepare details
            details = {
                "Waste Level (Classification)": prediction_class,
                "Circularity Score (Regression)": round(pred_reg, 2),
                "Material Recovery Rate": round(material_rate, 3),
                "Machine Downtime Ratio": round(downtime_ratio, 3)
            }

            # Recommendations
            recommendations = []
            if prediction_class == "High Waste":
                recommendations.append("Increase recycled input materials to reduce waste output.")
                recommendations.append("Optimize production efficiency to reduce raw material usage.")
                recommendations.append("Improve segregation at source to enhance material recovery rate.")
            else:
                recommendations.append("Maintain current waste efficiency — performance is good!")
                recommendations.append("Continue monitoring material flows to prevent future increases.")

            if material_rate < 0.4:
                recommendations.append("Material recovery rate is low — consider improving recycling processes.")

            if downtime_ratio > 0.3:
                recommendations.append("High machine downtime — consider maintenance scheduling optimization.")

        except Exception as e:
            details = {"Error": str(e)}

    return render_template(
        'index.html',
        sectors=sectors.keys(),
        details=details,
        recommendations=recommendations
    )


if __name__ == '__main__':
    app.run(debug=True)
