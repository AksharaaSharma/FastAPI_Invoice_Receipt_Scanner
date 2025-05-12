from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import re
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Setup
client = MongoClient("mongodb+srv://sharmaaa1604:FStsawGIXBx0Xpy5@cluster0.gaxdoul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["receipt_db"]
collection = db["receipts"]

class Receipt(BaseModel):
    items: list
    total: float
    vendor: str

@app.post("/upload/")
async def upload_receipt(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # OCR Processing
    text = pytesseract.image_to_string(image)
    
    # NLP Processing
    items = re.findall(r'(\d+\s?x?\s?.+?)\$?(\d+\.\d{2})', text)
    total = re.search(r'TOTAL\s+\$?(\d+\.\d{2})', text)
    
    # Store in MongoDB
    receipt_id = str(uuid.uuid4())
    document = {
        "_id": receipt_id,
        "items": [{"name": item[0], "price": item[1]} for item in items],
        "total": float(total.group(1)) if total else 0.0,
        "raw_text": text
    }
    collection.insert_one(document)
    
    return {"message": "Receipt processed", "id": receipt_id}

@app.get("/receipts/{receipt_id}")
async def get_receipt(receipt_id: str):
    return collection.find_one({"_id": receipt_id})
