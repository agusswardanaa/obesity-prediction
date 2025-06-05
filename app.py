import gradio as gr
import numpy as np
import joblib

# Create object model
gender_encoder = joblib.load("model/gender_encoder.pkl")
obesity_encoder = joblib.load("model/obesity_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/random_forest.joblib")

def predict(age, gender, height, weight, physical_activity_level):
    # Scaling numerical features
    numeric = scaler.transform([[age, height, weight]])
    gender_encoded = gender_encoder.transform([gender])
    activity = np.array([[physical_activity_level]])
    input = np.hstack([numeric, gender_encoded.reshape(-1,1), activity])

    # Predict
    prediction = model.predict(input)
    prediction = obesity_encoder.inverse_transform(prediction)
    
    return prediction[0]

demo = gr.Interface(
    fn=predict,
    title="Web Prediksi Tingkat Obesitas",
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(choices=["Male", "Female"], label="Gender"),
        gr.Number(label="Height"),
        gr.Number(label="Weight"),
        gr.Dropdown(choices=["1", "2", "3", "4"], label="Physical Activity Level")
    ],
    outputs=gr.Textbox(label="Hasil Prediksi")
)

if __name__ == "__main__":
    demo.launch(share=True)