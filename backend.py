from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

app = FastAPI()

# Enable CORS to allow requests from frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change ["http://localhost:5173"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model = load_model("densenet201_food_classification.h5")

# Define class indices
class_indices = {
    0: "burger",
    1: "butter_naan",
    2: "chai",
    3: "chapati",
    4: "chole_bhature",
    5: "dal_makhani",
    6: "dhokla",
    7: "fried_rice",
    8: "idli",
    9: "jalebi",
    10: "kaathi_rolls",
    11: "kadai_paneer",
    12: "kulfi",
    13: "masala_dosa",
    14: "momos",
    15: "paani_puri",
    16: "pakode",
    17: "pav_bhaji",
    18: "pizza",
    19: "samosa"
}

def predict_image(image, model):
    try:
        img = load_img(image, target_size=(224, 224))
        image_array = img_to_array(img) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        class_idx = np.argmax(predictions)
        class_label = class_indices.get(class_idx, "Unknown")
        confidence = float(predictions[0][class_idx])

        return class_label, confidence
    except Exception as e:
        return None, None

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = BytesIO(image_data)

        class_label, confidence = predict_image(image, model)

        if class_label is None:
            return {"error": "Prediction failed"}

        return {"predicted_class": class_label, "confidence": f"{confidence:.2f}"}
    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}
