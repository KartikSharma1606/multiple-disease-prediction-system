import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "medpredict_super_secret_clinical_key_2026")

# Directory configuration
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKING_DIR, "dataset")
MODELS_DIR = os.path.join(WORKING_DIR, "saved_models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DIABETES_CSV_PATH = os.path.join(DATASET_DIR, "diabetes.csv")
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "diabetes.pkl")
HEART_MODEL_PATH = os.path.join(MODELS_DIR, "heart.pkl")
KIDNEY_MODEL_PATH = os.path.join(MODELS_DIR, "kidney.pkl")

# 8 clinical parameter feature names for diabetes
DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ==========================================================
# RANDOM FOREST MODEL TRAINING PIPELINE FOR DIABETES
# ==========================================================
def train_diabetes_model(csv_path=DIABETES_CSV_PATH, model_save_path=DIABETES_MODEL_PATH):
    """
    Reads diabetes.csv, trains a Random Forest Classifier,
    evaluates its performance, and saves the serialized model pipeline.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Diabetes dataset not found at {csv_path}")

    print(f"[ML Pipeline] Loading diabetes dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    missing_cols = [col for col in DIABETES_FEATURES + ["Outcome"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    X = df[DIABETES_FEATURES]
    y = df["Outcome"]

    # Stratified train-test split for balanced representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[ML Pipeline] Training Random Forest Classifier on diabetes data...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Evaluation
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[ML Pipeline] Random Forest Model trained successfully. Test Accuracy: {acc * 100:.2f}%")

    # Serialize trained pipeline model
    with open(model_save_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"[ML Pipeline] Model saved to {model_save_path}")

    return rf_model


def load_or_train_diabetes_model():
    """
    Loads saved diabetes model; if missing or invalid, triggers the training pipeline.
    """
    if os.path.exists(DIABETES_MODEL_PATH):
        try:
            with open(DIABETES_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            # Verify model has predict and predict_proba
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                print("[ML Engine] Loaded existing diabetes model from saved_models/diabetes.pkl")
                return model
        except Exception as e:
            print(f"[ML Engine] Failed loading saved diabetes model ({e}). Retraining...")

    # If model is missing or incompatible, run the training pipeline
    return train_diabetes_model()


# Initialize models
diabetes_model = load_or_train_diabetes_model()

heart_model = None
if os.path.exists(HEART_MODEL_PATH):
    try:
        with open(HEART_MODEL_PATH, "rb") as f:
            heart_model = pickle.load(f)
        print("[ML Engine] Loaded Heart Disease model.")
    except Exception as e:
        print(f"[ML Engine] Warning: Could not load heart model: {e}")

kidney_model = None
if os.path.exists(KIDNEY_MODEL_PATH):
    try:
        with open(KIDNEY_MODEL_PATH, "rb") as f:
            kidney_model = pickle.load(f)
        print("[ML Engine] Loaded Kidney Disease model.")
    except Exception as e:
        print(f"[ML Engine] Warning: Could not load kidney model: {e}")


# ==========================================================
# FLASK ROUTES
# ==========================================================

@app.route("/")
def index():
    """Dashboard Overview page."""
    return render_template("index.html", active_page="dashboard")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    """
    Diabetes Prediction Route:
    - GET: renders the clinical input form with all 8 parameters.
    - POST: parses, validates 8 clinical parameters, predicts using Random Forest,
      computes risk probability percentage, and renders the result card.
    """
    prediction_text = None
    risk_percentage = None
    is_diabetic = False
    form_data = {}

    if request.method == "POST":
        try:
            # Capture all 8 clinical parameters (supporting both short and full field names)
            preg_raw = request.form.get("pregnancies") or request.form.get("Pregnancies")
            gluc_raw = request.form.get("glucose") or request.form.get("Glucose")
            bp_raw = request.form.get("blood_pressure") or request.form.get("BloodPressure")
            st_raw = request.form.get("skin_thickness") or request.form.get("SkinThickness")
            ins_raw = request.form.get("insulin") or request.form.get("Insulin")
            bmi_raw = request.form.get("bmi") or request.form.get("BMI")
            dpf_raw = (
                request.form.get("dpf")
                or request.form.get("DPF")
                or request.form.get("diabetes_pedigree_function")
                or request.form.get("DiabetesPedigreeFunction")
            )
            age_raw = request.form.get("age") or request.form.get("Age")

            # Retain user entries in form_data for easy correction
            form_data = {
                "pregnancies": preg_raw,
                "glucose": gluc_raw,
                "blood_pressure": bp_raw,
                "skin_thickness": st_raw,
                "insulin": ins_raw,
                "bmi": bmi_raw,
                "dpf": dpf_raw,
                "age": age_raw,
            }

            # Validate that all 8 parameters are present
            raw_values = [preg_raw, gluc_raw, bp_raw, st_raw, ins_raw, bmi_raw, dpf_raw, age_raw]
            if any(val is None or str(val).strip() == "" for val in raw_values):
                flash("All 8 clinical parameters are required. Please fill in every field.", "warning")
                return render_template("diabetes.html", active_page="diabetes", form_data=form_data)

            # Convert to float values
            pregnancies = float(preg_raw)
            glucose = float(gluc_raw)
            blood_pressure = float(bp_raw)
            skin_thickness = float(st_raw)
            insulin = float(ins_raw)
            bmi = float(bmi_raw)
            dpf = float(dpf_raw)
            age = float(age_raw)

            # Create DataFrame with exact column names expected by the model
            input_df = pd.DataFrame(
                [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                columns=DIABETES_FEATURES
            )

            # Predict using Random Forest model
            pred = diabetes_model.predict(input_df)[0]
            is_diabetic = bool(pred == 1)

            # Compute risk probability percentage
            if hasattr(diabetes_model, "predict_proba"):
                proba = diabetes_model.predict_proba(input_df)[0]
                # Index 1 corresponds to positive diabetes risk
                risk_percentage = round(float(proba[1]) * 100, 2)
            else:
                risk_percentage = 100.0 if is_diabetic else 0.0

            if is_diabetic:
                prediction_text = "The person is diabetic (High Risk Detected)"
            else:
                prediction_text = "The person is not diabetic (Low Risk / Healthy)"

        except ValueError as ve:
            flash(f"Invalid input: Please enter valid numerical values for all fields. ({ve})", "danger")
        except Exception as e:
            flash(f"An unexpected error occurred during prediction: {e}", "danger")

    return render_template(
        "diabetes.html",
        active_page="diabetes",
        prediction_text=prediction_text,
        risk_percentage=risk_percentage,
        is_diabetic=is_diabetic,
        form_data=form_data
    )


@app.route("/heart", methods=["GET", "POST"])
def heart():
    """Heart Disease Prediction Route."""
    prediction_text = None
    risk_percentage = None
    is_positive = False
    form_data = {}

    if request.method == "POST":
        try:
            fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            form_data = {f: request.form.get(f) for f in fields}

            if any(val is None or str(val).strip() == "" for val in form_data.values()):
                flash("All 13 cardiac parameters are required.", "warning")
                return render_template("heart.html", active_page="heart", form_data=form_data)

            input_values = [float(form_data[f]) for f in fields]

            if heart_model is None:
                flash("Heart Disease model is not currently loaded.", "danger")
                return render_template("heart.html", active_page="heart", form_data=form_data)

            # Predict using heart model
            pred = heart_model.predict([input_values])[0]
            is_positive = bool(pred == 1)

            if hasattr(heart_model, "predict_proba"):
                proba = heart_model.predict_proba([input_values])[0]
                risk_percentage = round(float(proba[1]) * 100, 2)
            else:
                risk_percentage = 100.0 if is_positive else 0.0

            if is_positive:
                prediction_text = "The person is having heart disease"
            else:
                prediction_text = "The person does not have any heart disease"

        except ValueError:
            flash("Please enter valid numerical values for all heart parameters.", "danger")
        except Exception as e:
            flash(f"Error during heart disease prediction: {e}", "danger")

    return render_template(
        "heart.html",
        active_page="heart",
        prediction_text=prediction_text,
        risk_percentage=risk_percentage,
        is_positive=is_positive,
        form_data=form_data
    )


@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    """Kidney Disease Prediction Route."""
    prediction_text = None
    risk_percentage = None
    is_positive = False
    form_data = {}

    fields = [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane"
    ]

    if request.method == "POST":
        try:
            form_data = {f: request.form.get(f) for f in fields}

            if any(val is None or str(val).strip() == "" for val in form_data.values()):
                flash("All 24 kidney parameters are required.", "warning")
                return render_template("kidney.html", active_page="kidney", form_data=form_data)

            input_values = [float(form_data[f]) for f in fields]

            if kidney_model is None:
                flash("Kidney Disease model is not currently loaded.", "danger")
                return render_template("kidney.html", active_page="kidney", form_data=form_data)

            pred = kidney_model.predict([input_values])[0]
            # In kidney dataset, classification target 0/1 depending on label encoding
            is_positive = bool(pred == 1 or pred == "ckd")

            if hasattr(kidney_model, "predict_proba"):
                proba = kidney_model.predict_proba([input_values])[0]
                pos_idx = 1 if len(proba) > 1 else 0
                risk_percentage = round(float(proba[pos_idx]) * 100, 2)

            if is_positive:
                prediction_text = "The person has chronic kidney disease"
            else:
                prediction_text = "The person does not have kidney disease"

        except ValueError:
            flash("Please enter valid numerical values for all kidney parameters.", "danger")
        except Exception as e:
            flash(f"Error during kidney disease prediction: {e}", "danger")

    return render_template(
        "kidney.html",
        active_page="kidney",
        prediction_text=prediction_text,
        risk_percentage=risk_percentage,
        is_positive=is_positive,
        form_data=form_data
    )


@app.route("/train-diabetes", methods=["POST"])
def trigger_train_diabetes():
    """Manual trigger endpoint to retrain the Random Forest model on diabetes.csv."""
    global diabetes_model
    try:
        diabetes_model = train_diabetes_model()
        flash("Random Forest Diabetes Model retrained successfully on diabetes.csv!", "success")
    except Exception as e:
        flash(f"Error retraining model: {e}", "danger")
    return redirect(url_for("diabetes"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
