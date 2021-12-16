from tensorflow import keras
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from fastapi import FastAPI, File
from PIL import Image
import io
import os

app = FastAPI()

models = {}

model = keras.models.load_model('model/model_b221b9f2-0eef-49bd-85c3-bae8d0f0be1f.h5')

@app.get("/")
async def root():
    return {"message": 'Everything is working kankerhond!'}

@app.post("/test")
async def create_file(file: bytes = File(...)):
    image = np.array(Image.open(io.BytesIO(file)))[:,:,:3]
    plt.imsave('image.png', image)
    image = np.array([image])

    pred = model.predict(image)

    if pred[0][0] > pred[0][1]:
        response = 'nut'
    else:
        response = 'bolt'

    return {"message": response}