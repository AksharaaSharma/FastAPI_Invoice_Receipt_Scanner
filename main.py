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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'combined_items' not in st.session_state:
    st.session_state.combined_items = []

@st.cache_resource
def initialize_paddle_ocr():
    """Initialize PaddleOCR with caching"""
    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
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

def process_image(image_file, paddle_ocr):
    """Process uploaded image file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name

        # Read and preprocess image
        img = cv2.imread(tmp_path)
        if img is None:
            return None

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 8)
        except Exception:
            bw = gray

        # OCR processing
        result = paddle_ocr.ocr(tmp_path, cls=True)
        
        # Cleanup temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass  # Ignore cleanup errors
        
        if not result or not result[0]:
            return None
            
        return "\n".join([line[1][0] for line in result[0]])
        
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def query_llm_groq(ocr_text, context, groq_api_key, max_retries=3):
    """Enhanced LLM query with context support - hardcoded API endpoint"""
    
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

    # Hardcoded model configurations - no environment variables
    model_configs = [
        {"model": "llama3-8b-8192", "temperature": 0.1, "max_tokens": 4096},
        {"model": "mixtral-8x7b-32768", "temperature": 0.1, "max_tokens": 4096},
        {"model": "llama3-70b-8192", "temperature": 0.1, "max_tokens": 4096}
    ]

    # Hardcoded API endpoint - no environment variables
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

    # Hardcoded categories - no external config
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
        
        # API Key input - direct input, no environment variables
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
            
            # Show preview of uploaded files
            with st.expander("📋 Preview Uploaded Files"):
                cols = st.columns(min(len(uploaded_files), 4))
                for idx, file in enumerate(uploaded_files[:4]):  # Show max 4 previews
                    with cols[idx % 4]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_container_width=True)
                
                if len(uploaded_files) > 4:
                    st.info(f"... and {len(uploaded_files) - 4} more files")

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
            results_container = st.container()
            
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
                    # OCR Processing
                    ocr_text = process_image(uploaded_file, paddle_ocr)
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
                        "data": structured_data
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Detailed Results", "📈 Visualization", "💾 Export"])
        
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
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Information:**")
                        data = result["data"]
                        st.write(f"• Invoice Number: {data.get('invoiceNumber', 'N/A')}")
                        st.write(f"• Date: {data.get('date', 'N/A')}")
                        st.write(f"• Total Amount: ${data.get('totalAmount', 0):.2f}")
                        st.write(f"• Purpose: {data.get('bill_purpose', 'Not specified')}")
                        
                        if result["context"]:
                            st.write(f"• Context: {result['context']}")
                    
                    with col2:
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
                    
                    # Raw JSON
                    with st.expander("View Raw JSON Data"):
                        st.json(data)

        with tab3:
            st.subheader("📊 Data Visualization")
            
            if st.session_state.combined_items:
                fig = create_visualization(st.session_state.combined_items)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No items found for visualization")
                
                # Category breakdown table
                st.subheader("📋 Category Breakdown")
                categories = {}
                for item in st.session_state.combined_items:
                    category = item.get("itemCategory", "others").lower()
                    if category not in categories:
                        categories[category] = {"count": 0, "total_value": 0}
                    categories[category]["count"] += 1
                    categories[category]["total_value"] += item.get("totalPrice", 0)
                
                category_df = pd.DataFrame([
                    {"Category": cat, "Item Count": data["count"], "Total Value": f"${data['total_value']:.2f}"}
                    for cat, data in categories.items()
                ])
                st.dataframe(category_df, use_container_width=True)
            else:
                st.info("No items available for visualization")

        with tab4:
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON Export
                if st.button("📄 Download JSON Results", use_container_width=True):
                    json_data = json.dumps(st.session_state.results, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="ocr_results.json",
                        mime="application/json"
                    )
            
            with col2:
                # CSV Export
                if st.button("📊 Download CSV Summary", use_container_width=True):
                    summary_data = []
                    for result in st.session_state.results:
                        data = result["data"]
                        summary_data.append({
                            "File": result["file"],
                            "Vendor": data.get("vendor", {}).get("name", "Unknown"),
                            "Date": data.get("date", "Unknown"),
                            "Invoice_Number": data.get("invoiceNumber", ""),
                            "Total_Amount": data.get("totalAmount", 0),
                            "Items_Count": len(data.get("items", [])),
                            "Purpose": data.get("bill_purpose", ""),
                            "Context": result.get("context", "")
                        })
                    
                    csv_data = pd.DataFrame(summary_data).to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="ocr_summary.csv",
                        mime="text/csv"
                    )
            
            # Items export
            if st.session_state.combined_items:
                st.subheader("📦 Export Items Data")
                items_df = pd.DataFrame(st.session_state.combined_items)
                csv_items = items_df.to_csv(index=False)
                st.download_button(
                    label="📦 Download All Items CSV",
                    data=csv_items,
                    file_name="all_items.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Preview items data
                with st.expander("Preview Items Data"):
                    st.dataframe(items_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Professional OCR Bill Processor | Built with Streamlit & PaddleOCR | Environment-Free Version"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
