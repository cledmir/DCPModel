from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model("model.h5")



@app.post("/predict")
async def predict(file: UploadFile):
    # Carga la imagen del archivo
    img = Image.open(file.file).convert("RGB")
    
    # Redimensiona la imagen a 224x224 y conviértela a un array de numpy
    img = img.resize((224, 224))
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
    return {"filename": file.filename, "prediction": prediction_labels[0], "Precisión": predictions[0][np.argmax(predictions, axis=1)][0]*100}

