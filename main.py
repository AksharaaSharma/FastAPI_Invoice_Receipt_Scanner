# main.py
import io
import re
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import pytesseract
from PIL import Image
import os

# Initialize FastAPI
app = FastAPI(
    title="Receipt Scanner API",
    description="API for processing receipts and invoices",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["receipt_db"]
collection = db["receipts"]

# Pydantic Models
class ReceiptItem(BaseModel):
    name: str
    price: float

class ReceiptResponse(BaseModel):
    id: str
    items: list[ReceiptItem]
    total: float
    vendor: Optional[str] = "Unknown"
    raw_text: str

@app.get("/", tags=["Health Check"])
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "active",
        "version": app.version,
        "docs": "/docs",
        "database_status": "connected" if client.server_info() else "disconnected"
    }

@app.post("/upload/", response_model=ReceiptResponse, status_code=status.HTTP_201_CREATED)
async def upload_receipt(file: UploadFile = File(...)):
    """Process receipt image/PDF and store extracted data"""
    try:
        # Read and verify file
        if not file.content_type.startswith(('image/', 'application/pdf')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported formats: JPEG, PNG, PDF"
            )

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # OCR Processing
        try:
            text = pytesseract.image_to_string(image)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"OCR processing failed: {str(e)}"
            )

        # NLP Processing (Enhanced regex patterns)
        items = re.findall(r'(\d+\s?x?\s?[\w\s]+?)\s+(\d+\.\d{2})', text)
        total_match = re.search(
            r'(?:total|balance|amount)\s+[\$]?(\d+\.\d{2})', 
            text, 
            re.IGNORECASE
        )
        
        # Generate document
        receipt_id = str(uuid.uuid4())
        document = {
            "_id": receipt_id,
            "items": [{"name": item[0].strip(), "price": float(item[1])} for item in items],
            "total": float(total_match.group(1)) if total_match else 0.0,
            "vendor": "Unknown",  # Add vendor detection logic as needed
            "raw_text": text
        }

        # MongoDB Insertion
        try:
            collection.insert_one(document)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database insertion failed: {str(e)}"
            )

        return {**document, "id": receipt_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/receipts/{receipt_id}", response_model=ReceiptResponse)
async def get_receipt(receipt_id: str):
    """Retrieve processed receipt by ID"""
    receipt = collection.find_one({"_id": receipt_id})
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return receipt

# Error Handling Middleware
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"},
    )
