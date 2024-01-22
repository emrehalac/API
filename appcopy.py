import os
##import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from cv2 import dnn_superres
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model dosyanızın adını belirtin
model_filename = "EDSR_x2.pb"

# Model dosyasının tam yolunu alın
model_path = os.path.join(os.path.dirname(__file__), model_filename)

# Eğer TensorFlow sürümü 2.x ise, try-except bloğu kullanarak yükleyin
    # Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)


    # Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)
    ##model = tf.saved_model.load(model_path)


@app.post("/upload/")
async def upload_photo(file: UploadFile = File(...)):
    # Dosya yolu
    file_path = os.path.join(os.path.dirname(__file__), "uploads", file.filename)

    # Yüklenen dosyayı geçici olarak bir klasörde tutun
    with open(file_path, "wb") as image:
        image.write(file.file.read())

    
    img = cv2.imread(file_path)
    #print(img.shape)
    # # Upscale the image
    result = sr.upsample(img)








    file_path_result = os.path.join(os.path.dirname(__file__), "uploads", "result_"+file.filename)
    cv2.imwrite(file_path_result,result)
    #print(result.shape)

   

    # Tahmin edilen çıktıyı kaydedin
    ##resized_file_path = os.path.join(os.path.dirname(__file__), "uploads", "resized_" + file.filename)
    # ...
    return FileResponse(file_path_result, media_type="image/jpeg")
    #return ("ok")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) #api'nin çalışacağı port ve host adresini yazıp çalıştırır
