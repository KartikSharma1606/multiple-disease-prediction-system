import os
import sys
import unittest
import pandas as pd
import numpy as np

# Import the Flask application and helper functions
from app import app, train_diabetes_model, load_or_train_diabetes_model, DIABETES_FEATURES

class TestMultiDiseaseApp(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_01_training_pipeline_and_csv(self):
        """Verify diabetes.csv exists and Random Forest model training pipeline works."""
        csv_path = os.path.join("dataset", "diabetes.csv")
        self.assertTrue(os.path.exists(csv_path), "dataset/diabetes.csv must exist")

        df = pd.read_csv(csv_path)
        for col in DIABETES_FEATURES + ["Outcome"]:
            self.assertIn(col, df.columns, f"Column {col} must exist in diabetes.csv")

        # Test training pipeline
        model = train_diabetes_model()
        self.assertIsNotNone(model, "train_diabetes_model should return a trained model")
        self.assertTrue(hasattr(model, "predict"), "Model must have predict method")
        self.assertTrue(hasattr(model, "predict_proba"), "Model must have predict_proba method")
        self.assertTrue(os.path.exists(os.path.join("saved_models", "diabetes.pkl")), "Model file should be saved")

    def test_02_diabetes_route_get(self):
        """Verify GET /diabetes renders diabetes.html with all 8 input fields."""
        response = self.client.get("/diabetes")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Check page heading
        self.assertIn("Diabetes", html)

        # Check all 8 clinical parameter inputs are present
        params = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "dpf", "age"]
        for p in params:
            self.assertIn(f'name="{p}"', html.lower(), f"Form must have input for {p}")

        # Check active state in sidebar
        self.assertIn('class="nav-item-link active"', html)
        self.assertIn("Diabetes Prediction", html)

    def test_03_diabetes_post_high_risk(self):
        """Verify POST /diabetes handles 8 clinical parameters and returns prediction & probability."""
        data = {
            "pregnancies": "6",
            "glucose": "148",
            "blood_pressure": "72",
            "skin_thickness": "35",
            "insulin": "0",
            "bmi": "33.6",
            "dpf": "0.627",
            "age": "50"
        }
        response = self.client.post("/diabetes", data=data)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Verify prediction result and risk percentage
        self.assertTrue("diabetic" in html.lower() or "risk" in html.lower())
        self.assertIn("Risk Probability", html)
        self.assertIn("%", html)

    def test_04_diabetes_post_low_risk(self):
        """Verify POST /diabetes handles low-risk parameters."""
        data = {
            "pregnancies": "1",
            "glucose": "85",
            "blood_pressure": "66",
            "skin_thickness": "29",
            "insulin": "0",
            "bmi": "26.6",
            "dpf": "0.351",
            "age": "31"
        }
        response = self.client.post("/diabetes", data=data)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        self.assertIn("The person is not diabetic", html)
        self.assertIn("Risk Probability", html)
        self.assertIn("%", html)

    def test_05_sidebar_navigation_and_active_states(self):
        """Verify sidebar navigation across all templates includes Diabetes, Heart, Kidney with active state."""
        routes = [
            ("/", "Dashboard Overview", "dashboard"),
            ("/diabetes", "Diabetes Prediction", "diabetes"),
            ("/heart", "Heart Disease Prediction", "heart"),
            ("/kidney", "Kidney Disease Prediction", "kidney")
        ]

        for route, title_text, active_key in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} should return 200")
            html = response.data.decode("utf-8")

            # Check that all 3 disease links exist in the sidebar
            self.assertIn("Diabetes Prediction", html)
            self.assertIn("Heart Disease Prediction", html)
            self.assertIn("Kidney Disease Prediction", html)

            # Check active state
            self.assertIn('class="nav-item-link active"', html)

    def test_06_validation_error_handling(self):
        """Verify validation handles invalid or missing inputs gracefully."""
        # Missing parameter
        data_missing = {
            "pregnancies": "1",
            "glucose": "85",
            "blood_pressure": "66"
            # others missing
        }
        resp = self.client.post("/diabetes", data=data_missing)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("All 8 clinical parameters are required", resp.data.decode("utf-8"))

        # Non-numeric input
        data_invalid = {
            "pregnancies": "one",
            "glucose": "normal",
            "blood_pressure": "70",
            "skin_thickness": "20",
            "insulin": "0",
            "bmi": "25",
            "dpf": "0.5",
            "age": "30"
        }
        resp = self.client.post("/diabetes", data=data_invalid)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Invalid input", resp.data.decode("utf-8"))

if __name__ == "__main__":
    unittest.main()
