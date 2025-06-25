import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd

st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="auto",
)

API_URL = "http://127.0.0.1:8000/predict/"

st.title("🍅 Tomato Disease Classifier")
st.write("Upload an image of a tomato leaf to classify its disease via API.")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if st.session_state.history:
    with st.expander("Show Prediction History"):
        history_data = [
            {
                "Image Name": item["filename"],
                "Predicted Disease": item["predicted_class"].replace('_', ' '),
                "Confidence": item["confidence"]
            }
            for item in st.session_state.history
        ]
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width =True)
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        predict_clicked = st.button('Predict Disease')
  

    if predict_clicked:
        with st.spinner('Sending image to API...'):
            image_bytes = uploaded_file.getvalue()
            files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    predicted_class = result.get("predicted_class", "N/A")
                    confidence = result.get("confidence", "N/A")
                    st.success(f"**Predicted Disease:** {predicted_class.replace('_', ' ')}")
                    st.info(f"**Confidence:** {confidence}")
                    # Save to history
                    st.session_state.history.append({
                        "filename": uploaded_file.name,
                        "image": image_bytes,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    })
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please make sure the backend server is running. Error: {e}")