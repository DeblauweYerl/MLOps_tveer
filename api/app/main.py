from tensorflow import keras
from fastapi import FastAPI, File

app = FastAPI()

model = keras.models.load_model('model/8522d1cc-40fc-4456-8ad2-4048dded0948.h5')

@app.post("/test")
async def create_file(file: bytes = File(...)):
    pred = model.predict(file)
    return {"message": pred}