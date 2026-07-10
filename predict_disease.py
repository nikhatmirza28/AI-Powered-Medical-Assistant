
# ==========================================
# AI-Powered Medical Assistant
# Disease Prediction Module
# ==========================================

import pickle
import pandas as pd

# ==========================================
# Load Model and Encoders
# ==========================================
print("Program Started")
model = pickle.load(open("disease_model.pkl", "rb"))
mlb = pickle.load(open("symptom_encoder.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load Dataset
df = pd.read_csv("medical_dataset.csv", keep_default_na=False)

# ==========================================
# Disease Prediction Function
# ==========================================

def predict_disease(user_symptoms):

    # -------------------------------
    # Clean User Input
    # -------------------------------

    cleaned_symptoms = []

    for symptom in user_symptoms:

        symptom = symptom.strip().lower()

        if symptom != "":
            cleaned_symptoms.append(symptom)

    # No symptoms entered
    if len(cleaned_symptoms) == 0:

        return {

            "Disease": "No Prediction",

            "Department": "-",

            "Severity": "-",

            "Advice": "Please enter valid symptoms.",

            "Prevention": "-"

        }

    # -------------------------------
    # Convert to ML Features
    # -------------------------------

    X_new = mlb.transform([cleaned_symptoms])

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(X_new)

    predicted_disease = label_encoder.inverse_transform(prediction)[0]

    # -------------------------------
    # Fetch Disease Details
    # -------------------------------

    disease_info = df[df["Disease"] == predicted_disease].iloc[0]

    # -------------------------------
    # Return Result
    # -------------------------------

    result = {

        "Disease": predicted_disease,

        "Department": disease_info["Department"],

        "Severity": disease_info["Severity"],

        "Advice": disease_info["Advice"],

        "Prevention": disease_info["Prevention"]

    }

    return result


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    symptoms = [
        "fever",
        "headache",
        "fatigue",
        "body pain"
    ]

    result = predict_disease(symptoms)

    print("\n========== Prediction ==========\n")

    for key, value in result.items():
        print(f"{key:<12}: {value}")

    print("\n===============================\n")


