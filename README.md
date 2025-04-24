# Disease-Prediction-
🩺 Disease Prediction Web Application
This is a Streamlit-powered web app that uses a trained XGBoost model (97% accuracy) to predict diseases based on clinical lab values. Designed for instant diagnosis, it simulates intelligent health screening and delivers interpretable insights, even for users without medical backgrounds.

① Project Objective
The aim of this project is to:

Assist in early disease detection using routine lab data

Provide risk stratification without invasive testing

Reduce dependency on manual diagnosis by using AI automation

Make medical interpretation accessible to general users

The model was trained and evaluated using Google Colab, where data preprocessing, EDA, and model evaluation were performed before integrating into a user-friendly Streamlit interface.

② Key Highlights
✅ Trained Model:

XGBoost algorithm fine-tuned on a dataset of 350,000+ entries

Achieved 97% accuracy, balancing precision and recall for all classes

Exported and used in Streamlit via .pkl files

🧹 Cleaned Dataset:

Processed for missing values and outliers

Feature correlation analysis and dimensionality checks

Used domain-specific logic to normalize values and impute unknowns

📊 Interactive UI:

Fully interactive prediction interface

Handles missing/unknown values with normal range random substitution

Real-time confidence-based prediction with natural language disease explanation

③ Medical Features Used
The model accepts 24 clinical parameters and gender:

Blood Composition: Hemoglobin, RBC, WBC, Platelets, Hematocrit

Liver Health: ALT, AST

Kidney & Cardiac: Creatinine, Troponin, C-Reactive Protein

Lipids & Glucose: HDL, LDL, Triglycerides, Glucose, Insulin

Vitals & Metabolic: BMI, Heart Rate, BP (Sys/Dia), HbA1c

Each parameter is mapped to gender-specific normal ranges, automatically used to substitute zeroes (unknowns) intelligently to ensure robust predictions even with partial data.

④ Streamlit Dashboard Features
4.1 🧠 Disease Predictor
Smart input system for male/female with default 0 substitution

All 24 features collected through clean 2-column layout

Button triggers cached prediction logic using XGBoost

Disease result is paired with detailed medical definition

4.2 📘 Normal Range Viewer
One-click access to merged male/female clinical range table

Second table explains each parameter's medical role

4.3 🤖 Gemini AI Chatbot
Conversational AI that answers disease-related questions

Works like a health assistant

Built-in memory of chat history

Uses Gemini API (Google Generative AI)

⑤ Predicted Diseases
The system can detect:

✅ Healthy

🩸 Anemia

❤️ Heart Attack Risk

🧬 Thalassemia

🔥 Liver Disease

⚠️ Hypertension

🧠 Coronary Artery Disease

💉 Hypercholesterolemia

💔 Kidney Disease

🧪 Thrombocytopenia

Each diagnosis is explained clearly, ensuring non-technical users can understand the result.


⑧ Screenshots & Previews
![home](https://github.com/user-attachments/assets/199dbeb8-478f-48e6-8674-3ff1548c0433)
![general ranges](https://github.com/user-attachments/assets/422a714c-880a-4555-8092-dc78e3a0cd1f)
![disease prediction](https://github.com/user-attachments/assets/32cb4ee7-f42c-457f-9e53-dde808bb9b15)
![chatbot](https://github.com/user-attachments/assets/18f3477f-9200-4a11-b294-9305426505e3)


