from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('best_logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('best_ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_sector.pkl', 'rb') as f:
    le_sector = pickle.load(f)

# Sectors displayed to user
sectors = ["Automotive", "Electronics", "Manufacturing", "FoodProcessing", "Textiles"]

@app.route('/', methods=['GET', 'POST'])
def index():
    details = None

    if request.method == "POST":
        try:
            # Get form inputs
            production_volume_units = float(request.form['production_volume_units'])
            raw_material_kg = float(request.form['raw_material_kg'])
            recycled_material_kg = float(request.form['recycled_material_kg'])
            waste_kg = float(request.form['waste_kg'])
            energy_kwh = float(request.form['energy_kwh'])
            water_liters = float(request.form['water_liters'])
            machine_downtime_hours = float(request.form['machine_downtime_hours'])
            sector_name = request.form['sector']

            # Derived features
            material_recovery_rate = recycled_material_kg / (recycled_material_kg + waste_kg + 1e-9)
            machine_downtime_ratio = (machine_downtime_hours * 60) / 1440

            # Sector Encoding (correct)
            sector_encoded = le_sector.transform([sector_name])[0]

            # Ordered feature vector (correct ordering!)
            X = np.array([
                production_volume_units,
                raw_material_kg,
                recycled_material_kg,
                waste_kg,
                energy_kwh,
                water_liters,
                material_recovery_rate,
                sector_encoded,
                machine_downtime_ratio
            ]).reshape(1, -1)

            # Scale only first 6 features
            X_scaled = X.copy()
            X_scaled[:, :6] = scaler.transform(X_scaled[:, :6])

            # Predictions
            class_pred_num = logistic_model.predict(X_scaled)[0]
            class_pred = "High waste" if class_pred_num == 1 else "Low waste"

            reg_pred = ridge_model.predict(X_scaled)[0]

            # Recommendations
            recommendations = []

            if class_pred_num == 1:
                recommendations.append("High waste detected — increase recycling processes.")
                recommendations.append("Optimize raw material usage to reduce waste.")
            else:
                recommendations.append("Waste level is low — maintain current efficiency.")
            
            # ----- Circularity Score Recommendations -----
            if reg_pred < 0.4:
                recommendations.append("Circularity score is low — prioritize recycling and material recovery.")
                recommendations.append("Consider adopting closed-loop material cycles to improve circularity.")
                recommendations.append("Evaluate product design for recyclability and durability.")
            elif reg_pred < 0.7:
                recommendations.append("Moderate circularity — improve by using more recycled materials.")
                recommendations.append("Enhance component reuse strategies to boost circularity further.")
            else:
                recommendations.append("Excellent circularity performance — sustain advanced recovery strategies.")
                recommendations.append("Document your best practices to maintain high circular performance.")

            if material_recovery_rate < 0.5:
                recommendations.append("Improve material recovery by increasing recycled inputs.")

            if machine_downtime_ratio > 0.3:
                recommendations.append("High downtime — schedule maintenance or improve machine reliability.")

            if energy_kwh > 10000:
                recommendations.append("High energy usage — consider energy-efficient machinery.")


            # Final output
            details = {
                "Waste Level (Classification)": class_pred,
                "Circularity Score (Regression)": round(reg_pred, 2),
                "Material Recovery Rate": round(material_recovery_rate, 3),
                "Machine Downtime Ratio": round(machine_downtime_ratio, 3),
                "Recommendations": recommendations
            }

        except Exception as e:
            details = {"Error": str(e)}

    return render_template("index.html", sectors=sectors, details=details)


if __name__ == '__main__':
    app.run(debug=True)
