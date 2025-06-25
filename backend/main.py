from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
import tensorflow as tf
from PIL import Image
import os
import json
import io
import numpy as np
from fastapi.responses import JSONResponse



MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tomato_disease_classifier(1).keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMAGE_SIZE = 128  

model = None
class_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        raise RuntimeError("Model or class names file not found. Make sure they are in the 'model/' directory.")
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    print(f"Loading class names from: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print("Class names loaded successfully:", class_names)
    yield

app = FastAPI(
    title="Tomato Disease Classifier API",
    description="An API to predict the disease of a tomato leaf from an image.",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file provided.")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")
        
    image_bytes = await file.read()
    
    preprocessed_image = preprocess_image(image_bytes)
    
    predictions = model.predict(preprocessed_image)
    score_array = predictions[0]  

    predicted_class_index = np.argmax(score_array)
    predicted_class = class_names[predicted_class_index]
    confidence_score = float(np.max(score_array)) * 100
    
    print(f"Prediction: {predicted_class}, Confidence: {confidence_score:.2f}%")
    
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": f"{confidence_score:.2f}%"
    })

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tomato Disease Classifier API. Use the /docs endpoint for more info."}