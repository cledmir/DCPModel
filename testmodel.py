from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Cargar modelo
model = load_model('model.h5')

# Cargar imagen de prueba
img = load_img('20230330_172242.jpg', target_size=(600, 450))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Realizar predicción en la imagen de prueba
preds = model.predict(x)

print(preds)