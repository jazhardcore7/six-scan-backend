import io
import base64
import re
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="Six Scan Backend")

MODEL_PATH = "model/yolov11n.pt"

# setup YOLO model dan EasyOCR
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

reader = easyocr.Reader(['en', 'id'], gpu=False) 

# --- Helper Functions ---

def parse_nutrition_data(text_list):
    # Default values jika tidak ditemukan
    data = {
        "energi": "0",
        "lemak": "0",
        "protein": "0",
        "karbohidrat": "0",
        "gula": "0",
        "garam": "0"
    }

    # Gabung teks jadi satu string agar regex bisa cari pattern multiline
    full_text = "\n".join(text_list)
    
    # Helper simpel buat ambil value dari regex
    def extract_value(pattern, text, default="0"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return default

    # 1. Energi (ambil angka + satuan, misal 100 kkal)
    energi_match = re.search(r'(?:Energi|Energy).*?(\d+(?:\.\d+)?\s*(?:kkal|kcal|kJ))', full_text, re.IGNORECASE | re.DOTALL)
    if energi_match:
        data["energi"] = energi_match.group(1)
    else:
        # Fallback kalau unit ga kebaca, ambil angkanya aja
        energi_match_num = re.search(r'(?:Energi|Energy).*?(\d+(?:\.\d+)?)', full_text, re.IGNORECASE | re.DOTALL)
        if energi_match_num:
             data["energi"] = energi_match_num.group(1)

    # 2. Lemak
    lemak_match = re.search(r'(?:Lemak|Fat).*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if lemak_match:
        data["lemak"] = lemak_match.group(1)
    
    # 3. Protein
    protein_match = re.search(r'Protein.*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if protein_match:
        data["protein"] = protein_match.group(1)

    # 4. Karbohidrat
    karbo_match = re.search(r'(?:Karbohidrat|Carb).*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if karbo_match:
        data["karbohidrat"] = karbo_match.group(1)

    # 5. Gula
    gula_match = re.search(r'(?:Gula|Sugar).*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if gula_match:
        data["gula"] = gula_match.group(1)

    # 6. Garam / Natrium
    garam_match = re.search(r'(?:Garam|Natrium|Salt|Sodium).*?(\d+(?:\.\d+)?\s*(?:mg|g))', full_text, re.IGNORECASE | re.DOTALL)
    if garam_match:
        data["garam"] = garam_match.group(1)

    return data

# --- Endpoints ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Baca image file
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        # Convert ke BGR karena OpenCV pake BGR
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Deteksi pakai YOLO
    results = model(image)
    
    # Ambil box dengan confidence tertinggi
    best_box = None
    max_conf = -1.0
    result = results[0]
    
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf > max_conf:
            max_conf = conf
            best_box = box.xyxy[0].tolist() # [x1, y1, x2, y2]

    # Kalau ga ada deteksi, return 404
    if best_box is None:
        return JSONResponse(content={"error": "No nutrition label detected"}, status_code=404)

    # Crop bagian label
    x1, y1, x2, y2 = map(int, best_box)
    
    h, w, _ = img_np.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped_img = img_np[y1:y2, x1:x2]
    
    if cropped_img.size == 0:
         return JSONResponse(content={"error": "Cropped area is empty"}, status_code=500)

    # Convert crop ke base64 buat dikirim balik ke frontend
    cropped_pil = Image.fromarray(cropped_img)
    buffered = io.BytesIO()
    cropped_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_response = f"data:image/jpeg;base64,{img_str}"

    # Proses OCR (Convert BGR dulu buat EasyOCR/OpenCV standard)
    cropped_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    ocr_result = reader.readtext(cropped_bgr, detail=0)

    # Parse data gizi
    nutrition_data = parse_nutrition_data(ocr_result)

    return {
        "cropped_image_base64": base64_response,
        "nutrition_data": nutrition_data,
        "ocr_raw": ocr_result 
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
