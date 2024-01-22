from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İzin verilecek kaynakları ayarlayabiliriz. Eğer domain'imiz olsa yani bir web uyg. yapsak buradan gelecek isteklere izin verir diğerleirni kapatabilirdik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the EDSR_x2 model
model_path = "C:\Users\emreh\Downloads\Bitirme Projesi\Bitirme Projesi\Bitirme Projesi (Model)\Modeller\EDSR"
print(model_path)
model = tf.saved_model.load(model_path)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}



@app.post("/upload/")
async def upload_photo(file: UploadFile = File(...)):
    # yüklenen dosyayı geçiçi olarak bi klasörde tutalım
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as image:
        image.write(file.file.read())

    # buraya modellerimizi gömmemiz gerekiyor
    # şuanlık yanıt görelim diye aynı şekilde yolladım
    resized_file_path = file_path

    return FileResponse(resized_file_path, media_type="image/jpeg") # Yine bize ŞUANLIK aynı resmi, daha sonrasında ise modellenmiş resmi dönecektir. 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) #api'nin çalışacağı port ve host adresini yazıp çalıştırır
