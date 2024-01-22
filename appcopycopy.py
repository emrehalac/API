import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from cv2 import dnn_superres
from typing import List

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

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel("edsr", 2)

async def process_single_image(img_path: str, output_folder: str) -> dict:
    img = cv2.imread(img_path)
    result = sr.upsample(img)

    result_filename = f"result_{os.path.basename(img_path)}"
    result_path = os.path.join(output_folder, result_filename)
    cv2.imwrite(result_path, result)

    return {
        "original_image": img_path,
        "enhanced_image": f"results/{result_filename}",  # Use a relative path
    }

@app.post("/process-folder/", response_model=List[dict])
async def process_folder(folder_path: str = "C:/Users/emreh/Desktop/modeldeneme"):
    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}

    output_folder = os.path.join(folder_path, "results")
    os.makedirs(output_folder, exist_ok=True)

    result_responses = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith((".jpg", ".png")):  # Add more extensions if needed
            img_path = os.path.join(folder_path, filename)
            result_response = await process_single_image(img_path, output_folder)
            result_responses.append(result_response)

    return FileResponse(file_path_result, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
