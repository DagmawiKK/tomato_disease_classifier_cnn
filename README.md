# Tomato Disease Classifier

This project provides a FastAPI-based backend and a Streamlit frontend for detecting whether an uploaded image is a tomato leaf and, if so, classifying its disease using deep learning models (YOLOv8 for detection and a custom CNN for disease classification).

---

## Features

- **Tomato/Non-Tomato Detection:** Uses a YOLOv8 model to detect if the uploaded image contains a tomato leaf.
- **Disease Classification:** If a tomato leaf is detected, a CNN model classifies the disease.
- **REST API:** Easily integrate with other applications or a frontend.
- **Streamlit Frontend:** User-friendly web interface for uploading images and viewing results.
- **Modular Code:** Core logic is separated into reusable modules.

---

## Project Structure

```
backend/
│
├── main.py                  # FastAPI app entry point
├── yolov8n.pt               # (Optional) YOLOv8 model file
├── model/
│   ├── best.pt              # YOLOv8 weights for tomato detection
│   ├── class_names.json     # Class names for disease classifier
│   ├── final_tomato_classifier.h5 # (Optional) Keras model for tomato/non-tomato
│   └── tomato_disease_classifier.keras # Keras model for disease classification
│
├── core/
│   ├── is_tomato.py         # Tomato detection logic
│   └── disease_classifier.py# Disease classification logic
│
└── train/                   # (Optional) Training scripts and notebooks

frontend/
└── app.py                   # Streamlit frontend app
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd tomato_disease_classifier_cnn
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Required packages include:**  
- fastapi
- uvicorn
- tensorflow
- torch
- ultralytics
- pillow
- numpy
- streamlit
- requests
- pandas

### 3. Place your models

- Place your YOLOv8 weights (`best.pt`) and Keras model (`tomato_disease_classifier.keras`) in the `backend/model/` directory.
- Ensure `class_names.json` is present in the `backend/model/` directory.

---

## Running the API

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).  
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running the Frontend

```bash
cd ../frontend
streamlit run app.py
```

---

## API Usage

### **POST** `/predict/`

- **Description:** Upload an image to detect if it is a tomato leaf and classify its disease.
- **Request:** `multipart/form-data` with a file field named `file`.
- **Response:** JSON with predicted class and confidence.

**Example using `curl`:**
```bash
curl -X POST "http://localhost:8000/predict/" -F "file=@/path/to/your/image.jpg"
```

---

## Model Details

- **YOLOv8 (`best.pt`):** Used for tomato leaf detection.
- **CNN (`tomato_disease_classifier.keras`):** Used for disease classification.
- **Class names:** Loaded from `class_names.json`.

---

## Customization

- To retrain or update models, use the scripts and notebooks in the `backend/train/` directory.
- You can adjust detection/classification thresholds in `backend/main.py` as needed.

---

## License

MIT License (or your chosen license)

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)

---

**For questions or contributions, please open an issue or pull request.**