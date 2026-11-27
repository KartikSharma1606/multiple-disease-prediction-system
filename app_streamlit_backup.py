import os
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide",
    page_icon="🏥",
)

# Get the working directory of the main application
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models for Heart and Kidney
heart_model_path = os.path.join(working_dir, "saved_models", "heart.pkl")
kidney_model_path = os.path.join(working_dir, "saved_models", "kidney.pkl")

heart_model = pickle.load(open(heart_model_path, "rb"))
kidney_model = pickle.load(open(kidney_model_path, "rb"))

# Sidebar navigation using native Streamlit radio (no pyarrow/custom components)
with st.sidebar:
  st.title("Disease Prediction System")
  selected = st.radio(
      "Select Disease Prediction",
      ["Heart Disease Prediction", "Kidney Disease Prediction"],
  )

# ==========================================
# 1. HEART DISEASE PREDICTION PAGE
# ==========================================
if selected == "Heart Disease Prediction":
  st.title("Heart Disease Prediction using ML")

  col1, col2, col3 = st.columns(3)

  with col1:
    age = st.text_input("Age")
  with col2:
    sex = st.text_input("Sex (1 = male; 0 = female)")
  with col3:
    cp = st.text_input("Chest Pain types (0, 1, 2, 3)")
  with col1:
    trestbps = st.text_input("Resting Blood Pressure")
  with col2:
    chol = st.text_input("Serum Cholestoral in mg/dl")
  with col3:
    fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)")
  with col1:
    restecg = st.text_input("Resting Electrocardiographic results (0, 1, 2)")
  with col2:
    thalach = st.text_input("Maximum Heart Rate achieved")
  with col3:
    exang = st.text_input("Exercise Induced Angina (1 = yes; 0 = no)")
  with col1:
    oldpeak = st.text_input("ST depression induced by exercise")
  with col2:
    slope = st.text_input("Slope of the peak exercise ST segment (0, 1, 2)")
  with col3:
    ca = st.text_input("Major vessels colored by fluoroscopy (0-3)")
  with col1:
    thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

  heart_diagnosis = ""

  if st.button("Heart Disease Test Result"):
    try:
      user_input = [
          float(age),
          float(sex),
          float(cp),
          float(trestbps),
          float(chol),
          float(fbs),
          float(restecg),
          float(thalach),
          float(exang),
          float(oldpeak),
          float(slope),
          float(ca),
          float(thal),
      ]
      prediction = heart_model.predict([user_input])
      if prediction[0] == 1:
        heart_diagnosis = "The person is having heart disease"
      else:
        heart_diagnosis = "The person does not have any heart disease"
      st.success(heart_diagnosis)
    except ValueError:
      st.error("Please enter valid numerical values for all fields.")

# ==========================================
# 2. KIDNEY DISEASE PREDICTION PAGE
# ==========================================
if selected == "Kidney Disease Prediction":
  st.title("Kidney Disease Prediction using ML")

  col1, col2, col3, col4 = st.columns(4)

  with col1:
    age = st.text_input("Age")
  with col2:
    bp = st.text_input("Blood Pressure")
  with col3:
    sg = st.text_input("Specific Gravity")
  with col4:
    al = st.text_input("Albumin")
  with col1:
    su = st.text_input("Sugar")
  with col2:
    rbc = st.text_input("Red Blood Cells (1:normal, 0:abnormal)")
  with col3:
    pc = st.text_input("Pus Cell (1:normal, 0:abnormal)")
  with col4:
    pcc = st.text_input("Pus Cell clumps (1:present, 0:notpresent)")
  with col1:
    ba = st.text_input("Bacteria (1:present, 0:notpresent)")
  with col2:
    bgr = st.text_input("Blood Glucose Random")
  with col3:
    bu = st.text_input("Blood Urea")
  with col4:
    sc = st.text_input("Serum Creatinine")
  with col1:
    sod = st.text_input("Sodium")
  with col2:
    pot = st.text_input("Potassium")
  with col3:
    hemo = st.text_input("Hemoglobin")
  with col4:
    pcv = st.text_input("Packed Cell Volume")
  with col1:
    wc = st.text_input("White Blood Cell Count")
  with col2:
    rc = st.text_input("Red Blood Cell Count")
  with col3:
    htn = st.text_input("Hypertension (1:yes, 0:no)")
  with col4:
    dm = st.text_input("Diabetes Mellitus (1:yes, 0:no)")
  with col1:
    cad = st.text_input("Coronary Artery Disease (1:yes, 0:no)")
  with col2:
    appet = st.text_input("Appetite (1:good, 0:poor)")
  with col3:
    pe = st.text_input("Pedal Edema (1:yes, 0:no)")
  with col4:
    ane = st.text_input("Anemia (1:yes, 0:no)")

  kidney_diagnosis = ""

  if st.button("Kidney Disease Test Result"):
    try:
      user_input = [
          float(age),
          float(bp),
          float(sg),
          float(al),
          float(su),
          float(rbc),
          float(pc),
          float(pcc),
          float(ba),
          float(bgr),
          float(bu),
          float(sc),
          float(sod),
          float(pot),
          float(hemo),
          float(pcv),
          float(wc),
          float(rc),
          float(htn),
          float(dm),
          float(cad),
          float(appet),
          float(pe),
          float(ane),
      ]
      prediction = kidney_model.predict([user_input])
      if prediction[0] == 1:
        kidney_diagnosis = "The person has chronic kidney disease"
      else:
        kidney_diagnosis = "The person does not have kidney disease"
      st.success(kidney_diagnosis)
    except ValueError:
      st.error("Please enter valid numerical values for all fields.")