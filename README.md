# Six Scan Backend

This is the backend service for the **Six Scan** application, a nutrition detection app. It provides an API to analyze images of food products, detect nutrition labels, and extract nutritional information.

## Features

*   **Object Detection**: Uses **YOLOv11** to detect nutrition labels in images.
*   **Image Processing**: Automatically crops the detected label area.
*   **Text Extraction**: Uses **EasyOCR** to read text from the cropped label.
*   **Data Parsing**: Parses the extracted text to identify key nutritional values (Energy, Fat, Protein, Carbohydrates, Sugar, Salt) using Regex.
*   **API**: Built with **FastAPI** for high performance and easy documentation.

## Tech Stack

*   **Python 3.8+**
*   **FastAPI**: Web framework.
*   **Ultralytics YOLO**: Object detection model.
*   **EasyOCR**: Optical Character Recognition.
*   **Pillow (PIL)**: Image manipulation.
*   **OpenCV**: Image processing.

## Prerequisites

*   Python 3.8 or higher installed.
*   A GPU is recommended for faster inference (YOLO & EasyOCR), but it runs on CPU as well.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Backend
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Setup:**
    Ensure the YOLO model file is placed correctly.
    *   Place your trained model `yolov11n.pt` in the `model/` directory.
    *   Path: `model/yolov11n.pt`

## Usage

1.  **Start the server:**
    ```bash
    python main.py
    ```
    Or using uvicorn directly:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Access the API:**
    The server will start at `http://0.0.0.0:8000`.

3.  **API Documentation:**
    Go to `http://localhost:8000/docs` to see the interactive Swagger UI documentation and test the endpoints.

## API Endpoints

### `POST /predict`

Uploads an image to detect and extract nutrition info.

*   **Request Body**: `multipart/form-data`
    *   `file`: The image file to analyze.
*   **Response**: JSON object containing:
    *   `cropped_image_base64`: Base64 encoded string of the cropped nutrition label.
    *   `nutrition_data`: Dictionary of extracted values (energi, lemak, protein, karbohidrat, gula, garam).
    *   `ocr_raw`: Raw list of text detected (for debugging).

## Project Structure

```
Backend/
├── model/
│   └── yolov11n.pt       # YOLO model file
├── main.py               # Main FastAPI application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```
