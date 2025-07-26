from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

# Properly extract actual model if nested
if isinstance(model_bundle, dict):
    if "model" in model_bundle and hasattr(model_bundle["model"], "predict"):
        model = model_bundle["model"]
    else:
        raise ValueError("The loaded dictionary does not contain a valid model under the key 'model'.")
elif hasattr(model_bundle, "predict"):
    model = model_bundle
else:
    raise ValueError("Loaded object is not a valid model.")

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        df = pd.DataFrame([input_data])

        # Apply encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        prediction = model.predict(df)[0]
        result = "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
