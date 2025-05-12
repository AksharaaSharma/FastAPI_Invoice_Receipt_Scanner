from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
from pymongo import MongoClient
import uvicorn
import os

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with your MongoDB Atlas connection URI
MONGODB_URI = "mongodb+srv://sharmaaa1604:FStsawGIXBx0Xpy5@cluster0.gaxdoul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Optional: use environment variables instead of hardcoding credentials
# MONGODB_URI = os.environ.get("MONGODB_URI")

client = MongoClient(MONGODB_URI)
db = client["invoices"]  # Use or create the "invoices" database

@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    text = pytesseract.image_to_string(image)

    # Basic line parsing — replace with your NLP logic
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    data = {"raw_text": text, "lines": lines}
    db.records.insert_one(data)

    return {"status": "success", "extracted_lines": lines}

# Run with: uvicorn your_filename:app --reload

