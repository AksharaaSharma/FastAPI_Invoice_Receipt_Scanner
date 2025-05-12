# dashboard.py
import streamlit as st
import requests

API_URL = "https://fastapi-invoice-receipt-scanner.onrender.com/"  # From Render dashboard

st.title("Receipt Management Dashboard")

uploaded_file = st.file_uploader("Upload Receipt")
if uploaded_file:
    response = requests.post(f"{API_URL}/upload/", files={"file": uploaded_file})
    st.success(f"Receipt ID: {response.json()['id']}")

receipt_id = st.text_input("Search Receipt by ID")
if receipt_id:
    receipt = requests.get(f"{API_URL}/receipts/{receipt_id}").json()
    st.json(receipt)

