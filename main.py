import requests
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pyrebase
import os
from uvicorn import run


app = FastAPI()


origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

config = {
    "apiKey": "AIzaSyCwqnqhGalmMpLzYew2u2aHZ6Loi7j0Cqo",
    "authDomain": "dcp-tp2.firebaseapp.com",
    "databaseURL": "https://dcp-tp2.firebaseio.com",
    "projectId": "dcp-tp2",
    "storageBucket": "dcp-tp2.appspot.com",
    "messagingSenderId": "608385181374",
    "appId": "1:608385181374:web:576c24cad0e0d8916c60ef"
}
firebase = pyrebase.initialize_app(config)

storage = firebase.storage()

# Carga el modelo de TensorFlow
try:
    model = tf.keras.models.load_model("model.h5")
except:
    None
    
    

# Función para leer una imagen de Firebase
def read_image_from_firebase(url):
    # Descarga la imagen a una variable de bytes
    response = requests.get(url)
    img_bytes = response.content
    
    # Convierte la variable de bytes a un array de numpy
    img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return img

@app.post("/predict")
async def predict(url: str):
    # Lee la imagen de Firebase
    img = read_image_from_firebase(url)
    
    # Redimensiona la imagen a 224x224
    img = cv2.resize(img, (224, 224))
    
    # Convierte la imagen a un array de numpy
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normaliza los valores de píxeles
    img_array = img_array / 255.0
    
    # Realiza la predicción
    predictions = model.predict(img_array)
    
    # Decodifica las etiquetas
    labels = ["Queratosis actínicas","carcinoma de células basales","lesiones benignas similares a queratosis","dermatofibroma ", "melanoma", "nevo melanocítico", "vascular lesions"]
    prediction_labels = [labels[i] for i in np.argmax(predictions, axis=1)]
    
    # Devuelve la respuesta JSON
    return {"filename": url, "prediction": prediction_labels[0], "probability": predictions[0][np.argmax(predictions, axis=1)][0]*100}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)