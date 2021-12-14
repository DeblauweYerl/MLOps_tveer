from tensorflow import keras
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from fastapi import FastAPI, File
from PIL import Image
import io

app = FastAPI()

model = keras.models.load_model('model/8522d1cc-40fc-4456-8ad2-4048dded0948.h5')

@app.post("/test")
async def create_file(file: bytes = File(...)):
    image = np.array(Image.open(io.BytesIO(file)))
    image = resize(image, (64,64,3))
    plt.imsave('image.png', image)
    print(image.shape)
    image = np.array([image])
    print(image.shape)

    pred = model.predict(image)
    return {"message": image.shape}