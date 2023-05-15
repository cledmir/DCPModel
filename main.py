import requests
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# Carga el modelo de TensorFlow
try:
    model = tf.keras.models.load_model("model.h5")
except:
    None
    
    

# Función para leer una imagen de Firebase
def read_image(url):
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
    img = read_image(url)
    
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