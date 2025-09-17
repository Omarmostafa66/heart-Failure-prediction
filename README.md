# 🩺 Heart Failure Prediction Project

This project uses a machine learning model to predict mortality events based on patient clinical records. The project includes a training script that compares two models and saves the best one, and a Gradio web app for user-friendly interaction.


## 📂 Project Structure
│
├── app.py # Gradio interface
├── best_rf_model.pkl # Trained ML model
├── requirements.txt # Project dependencies
├── README.md # Project description and usage

---

## ⚙️ Features

- Compare five models (e.g., Random Forest , Logistic Regression , Decision Tree , XGBoost , svc)
- Automatically saves the better-performing model
- Clean and user-friendly Gradio interface
- Easy to deploy on Hugging Face Spaces

---

## 📌 Links

| Type            | Link                                                                 |
|-----------------|----------------------------------------------------------------------|
| 🔗 Hugging Face | [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/mora2004/HeartFailure ) |
| 🖥️ Presentation | [Project Slides](https://drive.google.com/file/d/1foLTy3oQ4pbHYLaTunvakqFvfxQ643yr/view?usp=sharing)                |

## 🧠 Model Details

- **Input features**: Age, Anaemia, High blood pressure, Creatinine phosphokinase, Ejection fraction, Platelets, Serum creatinine, Serum sodium, Sex, Smoking
- **Output**: 0 = Alive, 1 = Death
- **Dataset**: [Heart Failure Clinical Records Dataset - Kaggle/UCI](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)


---
title: Heart Failure
emoji: 👁
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.41.1
app_file: app.py
pinned: false
license: mit
short_description: heart failure
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
