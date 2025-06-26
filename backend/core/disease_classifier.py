import tensorflow as tf
import numpy as np
from fastapi import HTTPException
from .is_tomato import preprocess_image

def classify_disease(image_bytes: bytes, model, class_names, size=128) -> tuple[str, float]:
    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")
    
    preprocessed_image = preprocess_image(image_bytes, size)
    predictions = model.predict(preprocessed_image)
    score_array = predictions[0]
    # If model output is logits, apply softmax
    if not np.isclose(np.sum(score_array), 1.0):
        score_array = tf.nn.softmax(score_array).numpy()
    predicted_class_index = np.argmax(score_array)
    predicted_class = class_names[predicted_class_index]
    confidence_score = float(np.max(score_array)) * 100
    return predicted_class, confidence_score