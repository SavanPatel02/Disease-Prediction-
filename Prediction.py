import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random

# ===============================
# Load Models and Data
# ===============================
try:
    model = joblib.load("models/disease_model (2).pkl")
    label_encoder = joblib.load("models/label_encoder (1).pkl")
    df = pd.read_csv("data/merged_medical_data.csv")

    # Feature names and default mean values
    features = [
        'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets',
        'White Blood Cells', 'Red Blood Cells', 'Hematocrit', 'Mean Corpuscular Volume',
        'Mean Corpuscular Hemoglobin', 'Mean Corpuscular Hemoglobin Concentration',
        'Insulin', 'BMI', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Triglycerides', 'HbA1c', 'LDL Cholesterol', 'HDL Cholesterol', 'ALT', 'AST',
        'Heart Rate', 'Creatinine', 'Troponin', 'C-reactive Protein'
    ]

except Exception as e:
    st.error(f"⚠️ Error loading model or data: {e}")
    st.stop()

# ===============================
# Normal Ranges for Males & Females
# ===============================
normal_ranges_male = {
    'Glucose': (70, 100), 'Cholesterol': (0, 200), 'Hemoglobin': (13.8, 17.2),
    'Platelets': (150000, 450000), 'White Blood Cells': (4500, 11000),
    'Red Blood Cells': (4.7, 6.1), 'Hematocrit': (40.7, 50.3),
    'Mean Corpuscular Volume': (80, 100), 'Mean Corpuscular Hemoglobin': (27, 33),
    'Mean Corpuscular Hemoglobin Concentration': (32, 36), 'Insulin': (2, 25),
    'BMI': (18.5, 24.9), 'Systolic Blood Pressure': (110, 130), 'Diastolic Blood Pressure': (70, 80),
    'Triglycerides': (0, 150), 'HbA1c': (0, 5.7), 'LDL Cholesterol': (0, 100),
    'HDL Cholesterol': (40, 100), 'ALT': (7, 55), 'AST': (8, 48),
    'Heart Rate': (60, 100), 'Creatinine': (0.74, 1.35), 'Troponin': (0, 0.01), 'C-reactive Protein': (0, 1)
}

normal_ranges_female = {
    'Glucose': (70, 100), 'Cholesterol': (0, 200), 'Hemoglobin': (12.1, 15.1),
    'Platelets': (150000, 450000), 'White Blood Cells': (4500, 11000),
    'Red Blood Cells': (4.2, 5.4), 'Hematocrit': (36.1, 44.3),
    'Mean Corpuscular Volume': (80, 100), 'Mean Corpuscular Hemoglobin': (27, 33),
    'Mean Corpuscular Hemoglobin Concentration': (32, 36), 'Insulin': (2, 25),
    'BMI': (18.5, 24.9), 'Systolic Blood Pressure': (110, 130), 'Diastolic Blood Pressure': (70, 80),
    'Triglycerides': (0, 150), 'HbA1c': (0, 5.7), 'LDL Cholesterol': (0, 100),
    'HDL Cholesterol': (50, 100), 'ALT': (7, 45), 'AST': (8, 48),
    'Heart Rate': (60, 100), 'Creatinine': (0.59, 1.04), 'Troponin': (0, 0.01), 'C-reactive Protein': (0, 1)
}

# ===============================
# Helper Functions
# ===============================

# ===============================
# Disease Descriptions Dictionary
# ===============================
disease_descriptions = {
    "Healthy": "No signs of disease or abnormalities in the body.",
    "Liver Disease": "Damage to the liver affecting its normal function, including hepatitis, fatty liver, and cirrhosis.",
    "Hypercholesterolemia": "High levels of cholesterol in the blood, increasing the risk of heart disease.",
    "Thalassemia": "Inherited blood disorder causing the body to produce abnormal hemoglobin, leading to anemia.",
    "Hypertension": "High blood pressure that increases the risk of heart attack, stroke, and kidney failure.",
    "Heart Attack Risk": "Increased likelihood of a heart attack due to factors like high cholesterol, hypertension, and obesity.",
    "Anemia": "Low levels of red blood cells or hemoglobin, leading to fatigue and weakness.",
    "Kidney Disease": "Impaired kidney function, preventing waste elimination and fluid balance.",
    "Coronary Artery Disease": "Narrowing of coronary arteries reducing blood flow to the heart, causing chest pain or heart attack.",
    "Thrombocytopenia": "Low platelet count that leads to bleeding disorders and bruising."
}


def handle_missing_values(gender, data):
    """Replace zero values with random values from normal range for the selected gender."""
    ranges = normal_ranges_male if gender == "Male" else normal_ranges_female
    for i, value in enumerate(data):
        if value == 0:  # Replace 0 with a random value from the normal range
            min_val, max_val = ranges[features[i]]
            data[i] = round(random.uniform(min_val, max_val), 2)
    return data


def predict_disease(gender, *inputs):
    """Make disease prediction based on user inputs and show description."""
    try:
        # Check if all inputs are 0 or empty
        if all(value == 0 or value == "" for value in inputs):
            return "✅ Predicted Status: Healthy"

        # Replace 0 with random values from the normal range
        processed_data = handle_missing_values(gender, list(inputs))

        # Create DataFrame with inputs
        input_df = pd.DataFrame([processed_data], columns=features)

        # Encode Gender
        input_df['Gender_Male'] = 1 if gender == "Male" else 0

        # Ensure feature alignment
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Use hash for consistent prediction on the same input
        input_hash = hash(tuple(processed_data))
        if "last_input_hash" not in st.session_state:
            st.session_state.last_input_hash = None

        # If input is unchanged, return the last prediction
        if st.session_state.last_input_hash == input_hash:
            return st.session_state.last_prediction

        # Make prediction using the model
        prediction_encoded = model.predict(input_df)[0]

        # Decode predicted label to disease name
        predicted_disease = label_encoder.inverse_transform([int(round(prediction_encoded))])[0]

        # Get disease description if available
        disease_desc = disease_descriptions.get(predicted_disease, None)

        # Prepare final result with description or fallback message
        if disease_desc:
            result_message = f"✅ Predicted Disease: {predicted_disease}\n\n📝 Description: {disease_desc}"
        else:
            result_message = (
                f"✅ Predicted Disease: {predicted_disease}\n\n⚠️ Description not found. "
                "Please ask the chatbot for more details."
            )

            

        # Save the prediction and hash to maintain consistency
        st.session_state.last_input_hash = input_hash
        st.session_state.last_prediction = result_message

        return result_message
    except Exception as e:
        return f"❌ Error during prediction: {e}"
    
    



# ===============================
# Streamlit UI Setup
# ===============================
st.set_page_config(page_title="Disease Prediction Dashboard", layout="wide")

# Hide sidebar "app" title using CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"]::before {
        content: "";
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    selected_page = st.radio(
        "Navigation", ["🏠 Home", "📊 Disease Prediction", "📚 Normal Ranges", "💬 Chatbot"], index=0
    )

# ===============================
# Home Page
# ===============================
if selected_page == "🏠 Home":
    st.title("🏥 Welcome to the Disease Prediction Dashboard")
    st.write(
        """
        - Use this dashboard to predict diseases based on medical data.
        - Chat with our AI-powered assistant for health-related queries.
        - Get accurate predictions and helpful suggestions quickly.
        """
    )
    st.image("images/healthcare_dashboard.jpeg", use_container_width=True)
    st.markdown("---")
    st.markdown("🔗 **Created by Savan Patel** | 🎯 **Disease Prediction Dashboard**")

# ===============================
# Disease Prediction Page
# ===============================
elif selected_page == "📊 Disease Prediction":
    st.title("🔍 Disease Prediction")

    st.markdown("---")
    st.markdown("📢 **Note:** Enter 0 for unknown values. System will replace it with normal range values from the table.")
    st.markdown("---")

    # Gender selection
    gender = st.radio("Gender", ['Male', 'Female'], horizontal=True)

    # User Input Section - 2x2 Grid Layout
    st.subheader("📝 Enter Patient Details")
    columns = st.columns(2)

    inputs = []
    for i, feature in enumerate(features):
        col = columns[i % 2]  # Alternate between 2 columns
        value = col.number_input(f"{feature}", min_value=0.0, max_value=1000000.0, value=0.0)
        inputs.append(value)

    # ===============================
    # Prediction Button Logic with Caching
    # ===============================
    if st.button("🔮 Predict"):
        # Initialize session state for caching inputs and predictions
        if "last_inputs" not in st.session_state:
            st.session_state.last_inputs = None
            st.session_state.last_prediction = None

        # Prepare current input for comparison
        current_inputs = [gender] + inputs  # Include gender for checking

        # Check if current inputs match the previous inputs
        if current_inputs == st.session_state.last_inputs:
            result = st.session_state.last_prediction
        else:
            # Make a new prediction if inputs have changed
            result = predict_disease(gender, *inputs)
            st.session_state.last_inputs = current_inputs
            st.session_state.last_prediction = result

        # Display the result
        st.success(result)



    

# ===============================
# Normal Ranges Page
# ===============================
elif selected_page == "📚 Normal Ranges":
    st.title("📚 Normal Ranges for Males & Females")

    # Merge Male and Female Ranges for Display in 1x1 Table
    male_df = pd.DataFrame(normal_ranges_male).T
    male_df.columns = ["Male Min", "Male Max"]

    female_df = pd.DataFrame(normal_ranges_female).T
    female_df.columns = ["Female Min", "Female Max"]

    # Combine Male and Female Tables in One DataFrame
    merged_df = pd.concat([male_df, female_df], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df.rename(columns={"index": "Parameter"}, inplace=True)

    # ===============================
    # Full Width Table Without Scroll
    # ===============================
    st.markdown("### 📊 Combined Normal Ranges Table")
    st.markdown(
        merged_df.style.format({
            "Male Min": "{:.2f}", "Male Max": "{:.2f}",
            "Female Min": "{:.2f}", "Female Max": "{:.2f}"
        }).set_table_styles([
            {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
            {"selector": "th, td", "props": [("padding", "8px"), ("border", "1px solid #ddd"), ("text-align", "center")]},
            # {"selector": "th", "props": [("background-color", "#f4f4f4")]}
        ]).to_html(),
        unsafe_allow_html=True,
    )

    # ===============================
    # Column Descriptions
    # ===============================
    # st.markdown("### 📚 Column Descriptions")
    column_descriptions = {
        "Glucose": "Blood sugar level that measures glucose concentration.",
        "Cholesterol": "Total cholesterol level, including HDL and LDL.",
        "Hemoglobin": "Protein in red blood cells that carries oxygen.",
        "Platelets": "Blood cells that help in clotting.",
        "White Blood Cells": "Cells that fight infection and disease.",
        "Red Blood Cells": "Cells that carry oxygen throughout the body.",
        "Hematocrit": "Percentage of red blood cells in the blood.",
        "Mean Corpuscular Volume": "Average size of red blood cells.",
        "Mean Corpuscular Hemoglobin": "Average amount of hemoglobin per red blood cell.",
        "Mean Corpuscular Hemoglobin Concentration": "Average concentration of hemoglobin in RBCs.",
        "Insulin": "Hormone that regulates blood sugar levels.",
        "BMI": "Body Mass Index (Weight to height ratio).",
        "Systolic Blood Pressure": "Top number in blood pressure, pressure when heart beats.",
        "Diastolic Blood Pressure": "Bottom number in blood pressure, pressure when heart rests.",
        "Triglycerides": "Type of fat found in blood.",
        "HbA1c": "Average blood sugar levels over 3 months.",
        "LDL Cholesterol": "Low-density lipoprotein, 'bad' cholesterol.",
        "HDL Cholesterol": "High-density lipoprotein, 'good' cholesterol.",
        "ALT": "Enzyme that indicates liver health.",
        "AST": "Enzyme that indicates liver function.",
        "Heart Rate": "Number of heartbeats per minute.",
        "Creatinine": "Byproduct of muscle metabolism, indicates kidney function.",
        "Troponin": "Protein that indicates heart muscle damage.",
        "C-reactive Protein": "Marker for inflammation in the body."
    }

    # Convert Descriptions to DataFrame
    description_df = pd.DataFrame.from_dict(column_descriptions, orient='index', columns=["Description"])
    description_df.reset_index(inplace=True)
    description_df.rename(columns={"index": "Parameter"}, inplace=True)

    # ===============================
    # Full Width Column Descriptions Table
    # ===============================
    st.markdown("### 📚 Column Descriptions")
    st.markdown(
        description_df.style.set_table_styles([
            {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
            {"selector": "th, td", "props": [("padding", "8px"), ("border", "1px solid #ddd"), ("text-align", "left")]},
            # {"selector": "th", "props": [("background-color", "#f4f4f4")]}
        ]).to_html(),
        unsafe_allow_html=True,
    )


# ===============================
# Chatbot Page (Unchanged As Requested)
# ===============================
elif selected_page == "💬 Chatbot":
    st.title("💬 Chatbot")

    import google.generativeai as genai

    api_key = st.secrets["api_key"]
    # Set up Gemini API key
    genai.configure(api_key=api_key)

    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-2.0-flash")

    # ===============================
    # Initialize Chat History
    # ===============================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chatbot UI
    st.write("Ask me anything!")

    # User input for chatbot
    user_input = st.text_input("You:", "")

    if user_input:
        try:
            # Generate response using Gemini model
            response = model.generate_content(user_input)

            # Save conversation to session history
            st.session_state.chat_history.append({"user": user_input, "bot": response.text})

            # Display the current response
            st.text_area("Gemini:", response.text, height=200)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # ===============================
    # Display Chat History
    # ===============================
    if st.session_state.chat_history:
        st.subheader("📚 Chat History")
        for chat in st.session_state.chat_history[::-1]:  # Reverse to show latest first
            with st.expander(f"User: {chat['user']}", expanded=False):
                st.write(f"**Bot:** {chat['bot']}")

    # ===============================
    # Clear Chat History with Page Reload
    # ===============================
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("✅ Chat history cleared successfully!")
        st.experimental_rerun()
