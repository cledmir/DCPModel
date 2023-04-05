import pandas as pd

data = pd.read_csv('HAM10000_metadata.csv')

import os
import shutil

for index, row in data.iterrows():
    image_path = 'imagenes/' + row['image_id'] + '.jpg'
    if os.path.exists(image_path):
        class_folder = 'Dataset/' + row['dx']
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        shutil.copy(image_path, class_folder)
