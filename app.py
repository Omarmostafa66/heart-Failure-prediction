import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names in the correct order for the model
feature_names = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
    'sex', 'smoking', 'time'
]

def predict_death_event(age, anaemia, cpk, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    """
    Predicts the death event based on input features, handling potential data errors.
    """
    # Combine all arguments into a list
    args = [age, anaemia, cpk, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]

    # 1. Validate and convert inputs
    try:
        if any(arg is None or str(arg).strip() == "" for arg in args):
            raise ValueError("One or more fields are empty.")
        
        # Convert boolean checkbox inputs to int
        processed_args = []
        for arg in args:
            if isinstance(arg, bool):
                processed_args.append(int(arg))
            else:
                processed_args.append(float(arg))

    except (ValueError, TypeError) as e:
        error_message = f"Input Error: Please ensure all fields are filled with valid numbers. Details: {e}"
        return error_message, gr.Label(visible=False)

    # 2. Create DataFrame and make prediction
    try:
        input_data = pd.DataFrame([processed_args], columns=feature_names)
        
        # Get the columns that the scaler expects
        scaler_features = scaler.feature_names_in_
        input_data_to_scale = input_data[scaler_features]
        
        # Create a copy to avoid SettingWithCopyWarning
        input_data_scaled = input_data.copy()
        input_data_scaled[scaler_features] = scaler.transform(input_data_to_scale)

        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)
    except Exception as e:
        error_message = f"Model Prediction Error: {e}"
        return error_message, gr.Label(visible=False)

    # 3. Format the successful output
    if prediction[0] == 1:
        result = "At-Risk (High probability of death event)"
    else:
        result = "Not At-Risk (Low probability of death event)"

    probabilities = {
        "Not At-Risk": prediction_proba[0][0],
        "At-Risk": prediction_proba[0][1]
    }

    return result, gr.Label(value=probabilities, visible=True)


# --- Gradio UI Layout ---
# --- THEME UPDATED TO gr.themes.Glass() for a professional look ---
with gr.Blocks(theme=gr.themes.Glass(), title="Heart Failure Prediction") as app:
    gr.Markdown(
        """
        # 🩺 Heart Failure Prediction App
        Enter patient data to predict the likelihood of a mortality event during the follow-up period.
        """
    )

    # --- Accordion for Inputs to make the UI cleaner ---
    with gr.Accordion("Step 1: Enter Patient Data", open=True):
        with gr.Group():
            gr.Markdown("### Patient Demographics & Lifestyle")
            with gr.Row():
                age = gr.Slider(label="Age", minimum=40, maximum=95, step=1, value=60, scale=2)
                sex = gr.Radio(label="Sex", choices=[("Female", 0), ("Male", 1)], value=1, type="value", scale=1)
            with gr.Row():
                anaemia = gr.Checkbox(label="Anaemia", value=False)
                diabetes = gr.Checkbox(label="Diabetes", value=False)
                high_blood_pressure = gr.Checkbox(label="High Blood Pressure", value=False)
                smoking = gr.Checkbox(label="Smoking", value=False)

        with gr.Group():
            gr.Markdown("### Clinical Measurements")
            with gr.Row():
                ejection_fraction = gr.Slider(label="Ejection Fraction [%]", minimum=14, maximum=80, step=1, value=38)
                serum_creatinine = gr.Slider(label="Serum Creatinine [mg/dL]", minimum=0.5, maximum=9.4, step=0.1, value=1.1)
                serum_sodium = gr.Slider(label="Serum Sodium [mEq/L]", minimum=113, maximum=148, step=1, value=136)
            with gr.Row():
                cpk = gr.Number(label="Creatine Phosphokinase (CPK) [mcg/L]", value=582)
                platelets = gr.Number(label="Platelets [kiloplatelets/mL]", value=263358.03)
                time = gr.Number(label="Follow-up Period [days]", value=130, info="Number of days for patient follow-up.")

    # --- Prediction Button ---
    predict_btn = gr.Button("Predict Risk", variant="primary", scale=1)

    # --- Output Section ---
    with gr.Accordion("Step 2: View Prediction Results", open=True):
         with gr.Group():
            output_label = gr.Label(label="Prediction Result", scale=2)
            output_confidence = gr.Label(label="Prediction Confidence", visible=True, scale=1)


    # Define the click event
    predict_btn.click(
        fn=predict_death_event,
        inputs=[
            age, anaemia, cpk, diabetes, ejection_fraction,
            high_blood_pressure, platelets, serum_creatinine,
            serum_sodium, sex, smoking, time
        ],
        outputs=[output_label, output_confidence]
    )
    
    # --- Add Examples for easier testing ---
    gr.Examples(
        examples=[
            [65, True, 160, True, 25, False, 289000, 1.7, 136, 1, True, 10], # At-Risk example
            [50, False, 582, False, 38, False, 263358, 1.1, 136, 1, False, 285] # Not At-Risk example
        ],
        inputs=[
            age, anaemia, cpk, diabetes, ejection_fraction,
            high_blood_pressure, platelets, serum_creatinine,
            serum_sodium, sex, smoking, time
        ],
        outputs=[output_label, output_confidence],
        fn=predict_death_event,
        cache_examples=True
    )


# --- Launch the App ---
if __name__ == "__main__":
    app.launch()