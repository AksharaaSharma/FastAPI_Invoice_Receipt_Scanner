import streamlit as st
import requests

st.title("🧾 Invoice & Receipt Scanner")

uploaded_file = st.file_uploader("Upload invoice image/PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    with st.spinner("Analyzing..."):
        res = requests.post("https://fastapi-invoice-receipt-scanner.streamlit.app/upload/", files={"file": uploaded_file})
    if res.ok:
        st.success("Extracted Text")
        for line in res.json()["extracted_lines"]:
            st.write(line)
    else:
        st.error("Failed to process the file")

