import streamlit as st
import cv2
import json
import os
import re
import requests
import time
from paddleocr import PaddleOCR
import plotly.graph_objects as go
from PIL import Image
import tempfile
import zipfile
from io import BytesIO
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Professional OCR Bill Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #e2e3f1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
    
    .image-preview {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'combined_items' not in st.session_state:
    st.session_state.combined_items = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

@st.cache_resource
def initialize_paddle_ocr():
    """Initialize PaddleOCR with caching - fixed parameters"""
    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang='en',
            # Removed show_log parameter as it's not supported
        )
    except Exception as e:
        st.error(f"Failed to initialize PaddleOCR: {str(e)}")
        return None

def safe_extract_json(text):
    """Enhanced JSON extraction with advanced error recovery"""
    if not text:
        return {"error": "Empty response text"}

    # First attempt - try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        error_msg = str(e)
        error_pos = e.pos if hasattr(e, 'pos') else None
        original_text = text
    except Exception as e:
        error_msg = str(e)
        error_pos = None
        original_text = text

    # Second attempt - try to locate JSON object/array
    json_str = None
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        match = re.search(pattern, text)
        if match:
            json_str = match.group()
            break

    if not json_str:
        return {"error": "No JSON structure found", "text_sample": text[:200] + "..."}

    # Common JSON fixes
    fixes = [
        (r'([\{,])(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)', r'\1\2"\3"\4'),
        (r"'([^']+)'", r'"\1"'),
        (r',(\s*[}\]])', r'\1'),
        (r'"\s*"', r'", "'),
        (r'\\', r'\\\\'),
        (r':\s*([0-9]+)\s*([,\]}])', r': \1\2')
    ]

    for pattern, replacement in fixes:
        try:
            json_str = re.sub(pattern, replacement, json_str)
        except Exception:
            continue

    # Final attempt with fixes
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        error_context = ""
        if error_pos and len(json_str) > error_pos:
            start = max(0, error_pos - 20)
            end = min(len(json_str), error_pos + 20)
            error_context = f" near: '{json_str[start:end]}'"
        
        return {
            "error": f"JSON parsing failed after fixes: {str(e)}{error_context}",
            "position": error_pos,
            "text_sample": json_str[:200] + "...",
            "original_text": original_text[:200] + "..."
        }

def preprocess_image_for_ocr(image_array):
    """
    Preprocess image for better OCR results using OpenCV (no GUI)
    Returns the processed image array and displays preview in Streamlit
    """
    try:
        # Convert PIL to OpenCV format if needed
        if len(image_array.shape) == 3:
            # Convert RGB to BGR for OpenCV processing
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            opencv_image = image_array
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing techniques
        preprocessing_options = {
            "Original Grayscale": gray,
            "Adaptive Threshold": cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8
            ),
            "Gaussian Threshold": cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8
            ),
            "Gaussian Blur + Threshold": cv2.threshold(
                cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],
            "Median Filter": cv2.medianBlur(gray, 3),
            "Morphological Operations": cv2.morphologyEx(
                gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            )
        }
        
        return preprocessing_options
        
    except Exception as e:
        st.error(f"Image preprocessing error: {str(e)}")
        return {"Original": gray if 'gray' in locals() else image_array}

def display_image_preprocessing_options(image_file, file_name):
    """Display preprocessing options in Streamlit interface"""
    try:
        # Load image
        pil_image = Image.open(image_file)
        image_array = np.array(pil_image)
        
        # Get preprocessing options
        processed_images = preprocess_image_for_ocr(image_array)
        
        st.subheader(f"🔧 Preprocessing Options for {file_name}")
        
        # Select preprocessing method
        selected_preprocessing = st.selectbox(
            f"Select preprocessing method for {file_name}:",
            list(processed_images.keys()),
            key=f"preprocessing_{file_name}"
        )
        
        # Display original and selected preprocessing side by side
        # Fixed: Removed nested columns structure
        st.write("**Original Image:**")
        st.image(pil_image, caption="Original", use_container_width=True)
        
        st.write(f"**{selected_preprocessing}:**")
        processed_img = processed_images[selected_preprocessing]
        st.image(processed_img, caption=selected_preprocessing, use_container_width=True)
        
        # Store the selected preprocessing for this image
        st.session_state.processed_images[file_name] = {
            'method': selected_preprocessing,
            'image': processed_img,
            'original': image_array
        }
        
        return processed_img
        
    except Exception as e:
        st.error(f"Error in preprocessing display: {str(e)}")
        return None

def process_image(image_file, paddle_ocr, use_preprocessing=True, file_name=None):
    """Process uploaded image file with optional preprocessing"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name

        # If preprocessing is enabled and we have a processed image, use it
        if use_preprocessing and file_name and file_name in st.session_state.processed_images:
            processed_img = st.session_state.processed_images[file_name]['image']
            
            # Save processed image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as processed_tmp:
                cv2.imwrite(processed_tmp.name, processed_img)
                processed_tmp_path = processed_tmp.name
            
            # Use processed image for OCR
            result = paddle_ocr.ocr(processed_tmp_path, cls=True)
            
            # Cleanup processed temp file
            try:
                os.unlink(processed_tmp_path)
            except:
                pass
        else:
            # Use original image
            result = paddle_ocr.ocr(tmp_path, cls=True)
        
        # Cleanup original temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if not result or not result[0]:
            return None
            
        extracted_text = "\n".join([line[1][0] for line in result[0]])
        
        # Display extracted text in Streamlit
        if file_name:
            with st.expander(f"📄 Extracted Text from {file_name}"):
                st.text_area(
                    "OCR Result:", 
                    extracted_text, 
                    height=200, 
                    key=f"ocr_text_{file_name}"
                )
        
        return extracted_text
        
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def display_image_analysis(image_file, file_name):
    """Display image analysis and quality metrics in Streamlit - Fixed nesting issue"""
    try:
        pil_image = Image.open(image_file)
        image_array = np.array(pil_image)
        
        # Calculate image quality metrics
        height, width = image_array.shape[:2] if len(image_array.shape) > 1 else (image_array.shape[0], 1)
        
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Calculate sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Display metrics without nested columns
        st.write("**Image Quality Metrics:**")
        
        # Create metrics in a single row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Resolution", f"{width}x{height}")
        
        with metric_col2:
            st.metric("Sharpness", f"{sharpness:.1f}")
        
        with metric_col3:
            st.metric("Brightness", f"{brightness:.1f}")
        
        with metric_col4:
            st.metric("Contrast", f"{contrast:.1f}")
        
        # Quality warnings and recommendations
        warnings = []
        recommendations = []
        
        if sharpness < 100:
            warnings.append("⚠️ Low sharpness detected")
            recommendations.append("📸 Consider taking a sharper image")
            
        if brightness < 50:
            warnings.append("⚠️ Image may be too dark")
            recommendations.append("💡 Increase lighting or brightness")
        elif brightness > 200:
            warnings.append("⚠️ Image may be too bright")
            recommendations.append("🔆 Reduce lighting or exposure")
            
        if contrast < 30:
            warnings.append("⚠️ Low contrast detected")
            recommendations.append("⚡ Improve contrast")
        
        # Display warnings
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        # Display recommendations
        if recommendations:
            st.subheader("💡 Quality Recommendations:")
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.success("✅ Image quality looks good for OCR processing!")
            
    except Exception as e:
        st.error(f"Error in image analysis: {str(e)}")

def query_llm_groq(ocr_text, context, groq_api_key, max_retries=3):
    """Enhanced LLM query with context support"""
    
    if len(ocr_text) > 7000:
        ocr_text = ocr_text[:7000] + "...[truncated]"
    
    context_section = ""
    if context and context.strip():
        context_section = f"""
ADDITIONAL CONTEXT PROVIDED BY USER:
{context.strip()}

Use this context to better understand the purpose and nature of this receipt/bill.
"""
    
    prompt = f"""
Please analyze this receipt OCR text and return structured data in PERFECTLY VALID JSON format:
{ocr_text}
{context_section}
Important rules for the JSON response:
1. ALL property names MUST be in double quotes
2. Use ONLY double quotes for strings (no single quotes)
3. No trailing commas in arrays or objects
4. No comments in the JSON
5. All special characters must be properly escaped
6. Categorise the items into the following categories: 
"electronics", "stationary", "electrical appliances", "tools",
"cleaning supplies", "beauty products", "clothing and jewellery",
"food and beverages", "books", "furniture", "grocery", "medicines", "others"
7. If context is provided, use it to better categorize items and determine their purpose
8. Using the context, determine the purpose of the bill include it in the JSON

Required structure:
{{
  "invoiceNumber": "string",
  "date": "string",
  "context": "{context if context else 'No additional context provided'}",
  "vendor": {{
    "name": "string",
    "address": "string or null",
    "contactDetails": "string or null",
    "taxID": "string or null"
  }},
  "items": [
    {{
      "itemName": "string",
      "itemCategory": "string",
      "quantity": 1,
      "unitPrice": 0.0,
      "totalPrice": 0.0
    }}
  ],
  "subtotal": 0.0,
  "discounts": 0.0,
  "taxVAT": {{
    "rate": 0.0,
    "amount": 0.0
  }},
  "totalAmount": 0.0,
  "bill_purpose": "string"
}}

Return ONLY valid JSON, no additional text or explanations.
"""
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    model_configs = [
        {"model": "llama3-8b-8192", "temperature": 0.1, "max_tokens": 4096},
        {"model": "mixtral-8x7b-32768", "temperature": 0.1, "max_tokens": 4096},
        {"model": "llama3-70b-8192", "temperature": 0.1, "max_tokens": 4096}
    ]

    api_endpoint = "https://api.groq.com/openai/v1/chat/completions"

    for config in model_configs:
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON generator that ALWAYS returns perfectly valid JSON. "
                              "Extract receipt data and format as JSON. Use any provided context to enhance the analysis."
                },
                {"role": "user", "content": prompt}
            ],
            **config
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    json=data,
                    timeout=45
                )
                
                if response.status_code == 400:
                    break
                
                response.raise_for_status()
                result = response.json()
                
                if not result.get("choices"):
                    continue
                    
                content = result["choices"][0]["message"]["content"].strip()
                
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                parsed = safe_extract_json(content)
                
                if "error" not in parsed:
                    return parsed
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    continue
                time.sleep(2)

    return {"error": "Failed with all available models and attempts"}

def create_visualization(items):
    """Create category visualization"""
    if not items:
        return None

    categories = [
        "electronics", "stationary", "electrical appliances", "tools",
        "cleaning supplies", "beauty products", "clothing and jewellery",
        "food and beverages", "books", "furniture", "grocery", "medicines", "others"
    ]
    
    counts = {category: 0 for category in categories}
    
    for item in items:
        category = item.get("itemCategory")
        if category is None:
            category = "others"
        else:
            category = str(category).lower().strip()
        
        if not category or category not in counts:
            category = "others"
        counts[category] += 1

    # Filter out zero counts for cleaner visualization
    filtered_counts = {k: v for k, v in counts.items() if v > 0}
    
    if not filtered_counts:
        return None

    fig = go.Figure(
        data=[go.Bar(
            x=list(filtered_counts.keys()),
            y=list(filtered_counts.values()),
            marker_color='rgba(102, 126, 234, 0.8)',
            marker_line_color='rgba(102, 126, 234, 1.0)',
            marker_line_width=2
        )]
    )
    
    fig.update_layout(
        title={
            'text': "Item Categories Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Category",
        yaxis_title="Number of Items",
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(248,249,250,1)',
        paper_bgcolor='rgba(248,249,250,1)',
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=80, b=150),
        height=500
    )
    
    return fig

# Main App Interface
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Professional OCR Bill Processor</h1>
        <p>Upload your receipts and bills for intelligent data extraction and analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key for LLM processing",
            placeholder="Enter API key here..."
        )
        
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key to proceed")
            st.info("💡 Get your free API key from: https://console.groq.com/")
            st.stop()
        
        st.header("🔧 Processing Options")
        enable_preprocessing = st.checkbox(
            "Enable Image Preprocessing",
            value=True,
            help="Apply image preprocessing for better OCR results"
        )
        
        show_image_analysis = st.checkbox(
            "Show Image Quality Analysis",
            value=True,
            help="Display image quality metrics and recommendations"
        )
        
        st.header("📋 Context Options")
        context_option = st.radio(
            "How would you like to handle context?",
            ["No context", "Same context for all", "Individual context"],
            help="Context helps the AI better understand the purpose of your bills"
        )
        
        global_context = ""
        if context_option == "Same context for all":
            global_context = st.text_area(
                "Global Context",
                placeholder="Enter context for all bills (e.g., 'Office supplies for Q1 2024')",
                help="This context will be applied to all uploaded bills"
            )

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Bills")
        uploaded_files = st.file_uploader(
            "Choose bill images",
            accept_multiple_files=True,
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload one or more bill/receipt images for processing"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully")
            
            # Show preview and analysis of uploaded files
            with st.expander("📋 Image Preview & Analysis", expanded=True):
                for idx, file in enumerate(uploaded_files):
                    st.subheader(f"📄 {file.name}")
                    
                    # Display image
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_container_width=True)
                    
                    # Show image analysis if enabled
                    if show_image_analysis:
                        file.seek(0)  # Reset file pointer
                        display_image_analysis(file, file.name)
                    
                    # Show preprocessing options if enabled
                    if enable_preprocessing:
                        file.seek(0)  # Reset file pointer
                        display_image_preprocessing_options(file, file.name)
                    
                    st.markdown("---")

    with col2:
        st.header("📊 Processing Stats")
        
        if uploaded_files:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Files Ready", len(uploaded_files))
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.processing_complete:
            success_count = len([r for r in st.session_state.results if "error" not in r.get("data", {})])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Successfully Processed", success_count)
            st.markdown('</div>', unsafe_allow_html=True)

    # Individual context input
    individual_contexts = {}
    if uploaded_files and context_option == "Individual context":
        st.header("📝 Individual Context for Each Bill")
        
        with st.expander("Set Context for Each Bill", expanded=True):
            for file in uploaded_files:
                individual_contexts[file.name] = st.text_input(
                    f"Context for {file.name}",
                    key=f"context_{file.name}",
                    placeholder="Enter specific context for this bill..."
                )

    # Processing button
    if uploaded_files:
        st.header("🚀 Process Bills")
        
        if st.button("Start Processing", type="primary", use_container_width=True):
            # Initialize OCR
            paddle_ocr = initialize_paddle_ocr()
            if not paddle_ocr:
                st.error("❌ Failed to initialize OCR engine")
                st.stop()
            
            # Processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            combined_items = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
                
                # Determine context
                if context_option == "Same context for all":
                    context = global_context
                elif context_option == "Individual context":
                    context = individual_contexts.get(uploaded_file.name, "")
                else:
                    context = ""
                
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # OCR Processing with preprocessing option
                    ocr_text = process_image(
                        uploaded_file, 
                        paddle_ocr, 
                        use_preprocessing=enable_preprocessing,
                        file_name=uploaded_file.name
                    )
                    
                    if not ocr_text:
                        st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                        continue
                    
                    # LLM Processing
                    structured_data = query_llm_groq(ocr_text, context, groq_api_key)
                    
                    if "error" in structured_data:
                        st.error(f"❌ Processing failed for {uploaded_file.name}: {structured_data['error']}")
                        continue
                    
                    # Success
                    result = {
                        "file": uploaded_file.name,
                        "context": context,
                        "data": structured_data,
                        "ocr_text": ocr_text
                    }
                    all_results.append(result)
                    
                    if "items" in structured_data and isinstance(structured_data["items"], list):
                        combined_items.extend(structured_data["items"])
                    
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            # Update session state
            st.session_state.results = all_results
            st.session_state.combined_items = combined_items
            st.session_state.processing_complete = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing completed!")
            
            if all_results:
                st.balloons()
                st.success(f"🎉 Successfully processed {len(all_results)} out of {len(uploaded_files)} bills!")
            else:
                st.error("❌ No bills were successfully processed")

    # Results section
    if st.session_state.processing_complete and st.session_state.results:
        st.header("📈 Results & Analytics")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📋 Detailed Results", "📈 Visualization", "🔍 OCR Review", "💾 Export"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Bills", len(st.session_state.results))
            
            with col2:
                total_amount = sum(
                    result["data"].get("totalAmount", 0) 
                    for result in st.session_state.results 
                    if isinstance(result["data"].get("totalAmount"), (int, float))
                )
                st.metric("Total Amount", f"${total_amount:.2f}")
            
            with col3:
                st.metric("Total Items", len(st.session_state.combined_items))
            
            with col4:
                unique_vendors = len(set(
                    result["data"].get("vendor", {}).get("name", "Unknown")
                    for result in st.session_state.results
                ))
                st.metric("Unique Vendors", unique_vendors)
            
            # Recent bills summary
            st.subheader("📄 Bills Summary")
            summary_data = []
            for result in st.session_state.results:
                data = result["data"]
                summary_data.append({
                    "File": result["file"],
                    "Vendor": data.get("vendor", {}).get("name", "Unknown"),
                    "Date": data.get("date", "Unknown"),
                    "Total Amount": f"${data.get('totalAmount', 0):.2f}",
                    "Items Count": len(data.get("items", [])),
                    "Purpose": data.get("bill_purpose", "Not specified")
                })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        with tab2:
            st.subheader("🔍 Detailed Bill Information")
            
            for idx, result in enumerate(st.session_state.results):
                with st.expander(f"📄 {result['file']} - {result['data'].get('vendor', {}).get('name', 'Unknown Vendor')}"):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.write("**Basic Information:**")
                        data = result["data"]
                        # COMPLETION OF THE CODE - Add this to the end of your existing code

                        st.write(f"• Invoice Number: {data.get('invoiceNumber', 'N/A')}")
                        st.write(f"• Date: {data.get('date', 'N/A')}")
                        st.write(f"• Total Amount: ${data.get('totalAmount', 0):.2f}")
                        st.write(f"• Purpose: {data.get('bill_purpose', 'Not specified')}")
                        
                        if result.get('context'):
                            st.write(f"• Context: {result['context']}")
                    
                    with detail_col2:
                        st.write("**Vendor Information:**")
                        vendor = data.get("vendor", {})
                        st.write(f"• Name: {vendor.get('name', 'N/A')}")
                        st.write(f"• Address: {vendor.get('address', 'N/A')}")
                        st.write(f"• Contact: {vendor.get('contactDetails', 'N/A')}")
                        st.write(f"• Tax ID: {vendor.get('taxID', 'N/A')}")
                    
                    # Items table
                    if data.get("items"):
                        st.write("**Items:**")
                        items_df = pd.DataFrame(data["items"])
                        st.dataframe(items_df, use_container_width=True)
                    
                    # Financial breakdown
                    st.write("**Financial Breakdown:**")
                    fin_col1, fin_col2, fin_col3 = st.columns(3)
                    
                    with fin_col1:
                        st.metric("Subtotal", f"${data.get('subtotal', 0):.2f}")
                    
                    with fin_col2:
                        tax_info = data.get('taxVAT', {})
                        st.metric("Tax/VAT", f"${tax_info.get('amount', 0):.2f}")
                    
                    with fin_col3:
                        st.metric("Discounts", f"${data.get('discounts', 0):.2f}")

        with tab3:
            st.subheader("📊 Data Visualization")
            
            # Category distribution
            if st.session_state.combined_items:
                fig = create_visualization(st.session_state.combined_items)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No items available for visualization")
            
            # Spending by vendor
            vendor_spending = {}
            for result in st.session_state.results:
                vendor_name = result["data"].get("vendor", {}).get("name", "Unknown")
                amount = result["data"].get("totalAmount", 0)
                if isinstance(amount, (int, float)):
                    vendor_spending[vendor_name] = vendor_spending.get(vendor_name, 0) + amount
            
            if vendor_spending:
                st.subheader("💰 Spending by Vendor")
                vendor_fig = go.Figure(
                    data=[go.Pie(
                        labels=list(vendor_spending.keys()),
                        values=list(vendor_spending.values()),
                        hole=0.3
                    )]
                )
                vendor_fig.update_layout(
                    title="Spending Distribution by Vendor",
                    font=dict(family="Arial, sans-serif", size=12),
                    height=400
                )
                st.plotly_chart(vendor_fig, use_container_width=True)
            
            # Monthly spending trend (if dates are available)
            date_amounts = []
            for result in st.session_state.results:
                date_str = result["data"].get("date", "")
                amount = result["data"].get("totalAmount", 0)
                if date_str and isinstance(amount, (int, float)):
                    try:
                        # Try to parse various date formats
                        import datetime
                        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                            try:
                                date_obj = datetime.datetime.strptime(date_str, fmt)
                                date_amounts.append({"date": date_obj, "amount": amount})
                                break
                            except:
                                continue
                    except:
                        pass
            
            if len(date_amounts) > 1:
                st.subheader("📈 Spending Trend Over Time")
                date_amounts.sort(key=lambda x: x["date"])
                dates = [item["date"] for item in date_amounts]
                amounts = [item["amount"] for item in date_amounts]
                
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=dates,
                    y=amounts,
                    mode='lines+markers',
                    name='Spending',
                    line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
                    marker=dict(size=8)
                ))
                
                trend_fig.update_layout(
                    title="Spending Trend Over Time",
                    xaxis_title="Date",
                    yaxis_title="Amount ($)",
                    font=dict(family="Arial, sans-serif", size=12),
                    height=400
                )
                st.plotly_chart(trend_fig, use_container_width=True)

        with tab4:
            st.subheader("🔍 OCR Text Review")
            st.info("Review the extracted OCR text for accuracy")
            
            for result in st.session_state.results:
                with st.expander(f"📄 OCR Text from {result['file']}"):
                    st.text_area(
                        f"Extracted text from {result['file']}:",
                        result.get('ocr_text', 'No OCR text available'),
                        height=300,
                        key=f"review_ocr_{result['file']}"
                    )

        with tab5:
            st.subheader("💾 Export Options")
            
            # Prepare export data
            all_data = []
            for result in st.session_state.results:
                data = result["data"]
                base_info = {
                    "File Name": result["file"],
                    "Context": result.get("context", ""),
                    "Invoice Number": data.get("invoiceNumber", ""),
                    "Date": data.get("date", ""),
                    "Vendor Name": data.get("vendor", {}).get("name", ""),
                    "Vendor Address": data.get("vendor", {}).get("address", ""),
                    "Vendor Contact": data.get("vendor", {}).get("contactDetails", ""),
                    "Vendor Tax ID": data.get("vendor", {}).get("taxID", ""),
                    "Subtotal": data.get("subtotal", 0),
                    "Discounts": data.get("discounts", 0),
                    "Tax Rate": data.get("taxVAT", {}).get("rate", 0),
                    "Tax Amount": data.get("taxVAT", {}).get("amount", 0),
                    "Total Amount": data.get("totalAmount", 0),
                    "Bill Purpose": data.get("bill_purpose", "")
                }
                
                # Add items
                if data.get("items"):
                    for item in data["items"]:
                        row = base_info.copy()
                        row.update({
                            "Item Name": item.get("itemName", ""),
                            "Item Category": item.get("itemCategory", ""),
                            "Quantity": item.get("quantity", 0),
                            "Unit Price": item.get("unitPrice", 0),
                            "Item Total Price": item.get("totalPrice", 0)
                        })
                        all_data.append(row)
                else:
                    all_data.append(base_info)
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # CSV Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"bill_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # JSON Export
                json_data = {
                    "export_timestamp": pd.Timestamp.now().isoformat(),
                    "total_bills": len(st.session_state.results),
                    "bills": st.session_state.results
                }
                
                st.download_button(
                    label="📥 Download as JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"bill_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Excel Export
                try:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Bill Analysis', index=False)
                        
                        # Create summary sheet
                        summary_data = {
                            'Metric': ['Total Bills', 'Total Amount', 'Total Items', 'Unique Vendors'],
                            'Value': [
                                len(st.session_state.results),
                                sum(r["data"].get("totalAmount", 0) for r in st.session_state.results if isinstance(r["data"].get("totalAmount"), (int, float))),
                                len(st.session_state.combined_items),
                                len(set(r["data"].get("vendor", {}).get("name", "Unknown") for r in st.session_state.results))
                            ]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"bill_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.warning(f"Excel export not available: {str(e)}")
                
                # Preview data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                if len(df) > 10:
                    st.info(f"Showing first 10 rows. Total rows: {len(df)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔍 Professional OCR Bill Processor | Built with Streamlit & PaddleOCR</p>
        <p>Powered by Groq API for intelligent data extraction</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
