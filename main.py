from typing import Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("cnnModels/model28-Adam.keras")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            #Decodificar como imagen en escala de grises
            img_tensor = tf.image.decode_image(data, channels=1)
            img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)  # Escalar a [0, 1]
            #Redimensionar
            img_tensor = tf.image.resize(img_tensor, (48,48))
            #Expandir dimensión para batch
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            result = np.argmax(model.predict(img_tensor), axis=1)[0]
            print(result)
            await websocket.send_text(str(result))
    except WebSocketDisconnect:
        print("Cliente desconectado")