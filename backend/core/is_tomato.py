import tensorflow as tf
from PIL import Image
import io
import numpy as np
from fastapi import HTTPException
import tempfile

def preprocess_image(image_bytes: bytes, size: int) -> tf.Tensor:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((size, size))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

def is_tomato_leaf_yolo(image_bytes: bytes, yolo_model) -> tuple[bool, float]:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        results = yolo_model.predict(tmp.name)
        print("YOLO results:", results)
        found_tomato = False
        tomato_conf = 0.0
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id].lower()
            conf = float(box.conf[0])
            print(f"Detected class_id: {class_id}, class_name: {class_name}, conf: {conf:.2f}")
            if "tomato" in class_name:
                found_tomato = True
                tomato_conf = conf * 100
        return (found_tomato, tomato_conf) if found_tomato else (False, 0.0)