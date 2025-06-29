from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
import tensorflow as tf
import os
import json
from fastapi.responses import JSONResponse
import torch.nn.modules.container
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils import IterableSimpleNamespace
import torch.serialization
from ultralytics.nn.modules import (
    Conv, C2f, SPPF, Bottleneck, Concat, Detect, DFL
)
from torch.nn.modules.activation import SiLU
import torch.nn.modules.conv
import torch.nn.modules.batchnorm
import torch.nn.modules.pooling
import torch.nn.modules.upsampling
from core.is_tomato import is_tomato_leaf_yolo
from core.disease_classifier import classify_disease
import tempfile
import shutil

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tomato_disease_classifier(1).keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
TOMATO_MODEL_PATH = os.path.join(MODEL_DIR, "final_tomato_classifier.h5")
IMAGE_SIZE = 128

model = None
class_names = None
tomato_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, tomato_model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH) or not os.path.exists(TOMATO_MODEL_PATH):
        raise RuntimeError("Model or class names file not found. Make sure they are in the 'model/' directory.")
    print(f"Loading disease model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Disease model loaded successfully.")
    print(f"Loading class names from: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print("Class names loaded successfully:", class_names)
    print(f"Loading tomato/non-tomato model from: {TOMATO_MODEL_PATH}")
    tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)
    print("Tomato/non-tomato model loaded successfully.")
    yield

app = FastAPI(
    title="Tomato Disease Classifier API",
    description="An API to predict the disease of a tomato leaf from an image.",
    version="1.0.0",
    lifespan=lifespan,
)

with torch.serialization.safe_globals([
    DetectionModel,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.container.Sequential,
    Conv,
    C2f,
    SPPF,
    Bottleneck,
    Concat,
    Detect,
    DFL,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    SiLU,
    IterableSimpleNamespace,
]):
    yolo_model = YOLO(os.path.join(MODEL_DIR, "best.pt"))
yolo_model.overrides['conf'] = 0.25
yolo_model.overrides['iou'] = 0.45
yolo_model.overrides['agnostic_nms'] = False
yolo_model.overrides['max_det'] = 1000

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # YOLO tomato leaf check
    is_tomato, tomato_conf = is_tomato_leaf_yolo(image_bytes, yolo_model)
    if not is_tomato:
        return JSONResponse(content={
            "predicted_class": "Not a tomato leaf",
            "confidence": f"{tomato_conf:.2f}%"
        })
    # Disease classification
    predicted_class, confidence_score = classify_disease(image_bytes, model, class_names)
    
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": f"{confidence_score:.2f}%"
    })

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tomato Disease Classifier API. Use the /docs endpoint for more info."}