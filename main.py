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

# --- Configuration ---
MODEL_PATH = "model/yolov11n.pt"
# Initialize YOLO model
# We load it globally to avoid reloading on every request
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Initialize EasyOCR reader
# GPU=True if available, else False.
# We'll use English and Indonesian (since keywords like 'Lemak', 'Gula' are Indonesian)
reader = easyocr.Reader(['en', 'id'], gpu=False) # Set gpu=True if you have a GPU configured

# --- Helper Functions ---

def parse_nutrition_data(text_list):
    """
    Parses a list of strings from OCR to find nutrition values.
    Returns a dictionary with keys: energi, lemak, protein, karbohidrat, gula, garam.
    """
    data = {
        "energi": "0",
        "lemak": "0",
        "protein": "0",
        "karbohidrat": "0",
        "gula": "0",
        "garam": "0"
    }

    # Join all text to make regex search easier across potential line breaks if needed,
    # but searching line by line or in a joined block depends on OCR quality.
    # Let's try searching in the whole text block first for robustness, 
    # or iterate through lines. Given OCR often splits label and value, 
    # a joined string might be better if we handle newlines.
    
    # However, simple regex on the list is often safer to avoid false positives from far away text.
    # Let's try a hybrid approach: Join with newlines.
    full_text = "\n".join(text_list)
    
    # Helper to extract value based on pattern
    def extract_value(pattern, text, default="0"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Return the captured number and unit if present, or just number
            # The regex groups usually capture the number.
            # Let's assume we want the full string like "100 kkal" or just "100".
            # The prompt asks for "Extract number + unit" for Energi.
            return match.group(1).strip()
        return default

    # 1. Energi Total (Keywords: Energi, Energy) -> Number + unit
    # Pattern: (Energi|Energy).*?(\d+(?:\.\d+)?\s*(?:kkal|kcal|kJ))
    # Note: .*? matches non-greedy.
    energi_match = re.search(r'(?:Energi|Energy).*?(\d+(?:\.\d+)?\s*(?:kkal|kcal|kJ))', full_text, re.IGNORECASE | re.DOTALL)
    if energi_match:
        data["energi"] = energi_match.group(1)
    else:
        # Fallback: sometimes just number
        energi_match_num = re.search(r'(?:Energi|Energy).*?(\d+(?:\.\d+)?)', full_text, re.IGNORECASE | re.DOTALL)
        if energi_match_num:
             data["energi"] = energi_match_num.group(1)

    # For other nutrients, we usually just want the number, but let's keep the unit if it's close.
    # The prompt implies returning the value found.
    
    # 2. Lemak Total (Keywords: Lemak, Fat)
    lemak_match = re.search(r'(?:Lemak|Fat).*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if lemak_match:
        data["lemak"] = lemak_match.group(1)
    
    # 3. Protein (Keywords: Protein)
    protein_match = re.search(r'Protein.*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if protein_match:
        data["protein"] = protein_match.group(1)

    # 4. Karbohidrat Total (Keywords: Karbohidrat, Carb)
    karbo_match = re.search(r'(?:Karbohidrat|Carb).*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if karbo_match:
        data["karbohidrat"] = karbo_match.group(1)

    # 5. Gula (Keywords: Gula, Sugar)
    gula_match = re.search(r'(?:Gula|Sugar).*?(\d+(?:\.\d+)?\s*g)', full_text, re.IGNORECASE | re.DOTALL)
    if gula_match:
        data["gula"] = gula_match.group(1)

    # 6. Garam (Keywords: Garam, Natrium, Salt, Sodium)
    garam_match = re.search(r'(?:Garam|Natrium|Salt|Sodium).*?(\d+(?:\.\d+)?\s*(?:mg|g))', full_text, re.IGNORECASE | re.DOTALL)
    if garam_match:
        data["garam"] = garam_match.group(1)

    return data

# --- Endpoints ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Read Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Convert to numpy array for OpenCV/YOLO
        img_np = np.array(image)
        # OpenCV uses BGR, but YOLO/PIL use RGB. Ultralytics handles RGB PIL images fine.
        # But for cropping and encoding back to base64 via OpenCV, we might need BGR.
        # Let's stick to PIL for cropping to avoid color confusion, or just be careful.
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. YOLO Detection
    results = model(image)
    
    # Find the "Nutrition Label" class. 
    # We assume the model is trained to detect this class.
    # If the model has multiple classes, we should filter by class name or ID.
    # For now, we take the detection with the highest confidence.
    
    best_box = None
    max_conf = -1.0

    # results is a list (one per image), we only have one image
    result = results[0]
    
    for box in result.boxes:
        # box.conf is a tensor, box.xyxy is a tensor
        conf = float(box.conf[0])
        # Check class if necessary: int(box.cls[0])
        # Assuming class 0 is Nutrition Label or we just want the most confident box of any class
        if conf > max_conf:
            max_conf = conf
            best_box = box.xyxy[0].tolist() # [x1, y1, x2, y2]

    if best_box is None:
        return JSONResponse(content={"error": "No nutrition label detected"}, status_code=404)

    # 3. Cropping
    x1, y1, x2, y2 = map(int, best_box)
    
    # Ensure boundaries
    h, w, _ = img_np.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped_img = img_np[y1:y2, x1:x2]
    
    if cropped_img.size == 0:
         return JSONResponse(content={"error": "Cropped area is empty"}, status_code=500)

    # Convert cropped image to Base64
    # Use PIL to save to memory buffer
    cropped_pil = Image.fromarray(cropped_img)
    buffered = io.BytesIO()
    cropped_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_response = f"data:image/jpeg;base64,{img_str}"

    # 4. OCR Processing
    # EasyOCR expects a file path, numpy array (BGR), or bytes.
    # We have cropped_img which is RGB (from img_np). EasyOCR uses BGR if using OpenCV backend, 
    # but it handles RGB numpy arrays too? Actually EasyOCR docs say: 
    # "The input image can be a file path, a numpy array (BGR), or a byte stream."
    # Let's convert to BGR to be safe as that's standard OpenCV format.
    cropped_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    
    ocr_result = reader.readtext(cropped_bgr, detail=0) # detail=0 returns just the list of strings

    # 5. Data Parsing
    nutrition_data = parse_nutrition_data(ocr_result)

    # 6. Response
    return {
        "cropped_image_base64": base64_response,
        "nutrition_data": nutrition_data,
        "ocr_raw": ocr_result # Optional: for debugging
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
