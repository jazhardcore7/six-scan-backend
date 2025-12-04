import requests
from PIL import Image
import io

# Create a dummy image (black square)
img = Image.new('RGB', (640, 640), color = 'black')
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)

url = "http://localhost:8000/predict"
files = {'file': ('test.jpg', buf, 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure it is running.")
