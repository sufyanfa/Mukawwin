import io
import os
import json
import streamlit as st
from google.cloud import vision
import google.generativeai as genai

# --- UI Setup ---
st.set_page_config(page_title="🔍 Ingredient Extractor", layout="centered")
st.title("🤖 Smart Ingredient Extractor")
st.write("Upload an image of a food product, and the system will automatically extract the ingredients list.")

# --- API Configuration ---
# Secure way to handle keys using Streamlit secrets for deployment
try:
    # Attempt to load from Streamlit secrets first
    google_creds_json_str = st.secrets.get("gcp_service_account_json_str")
    gemini_api_key = st.secrets.get("gemini_api_key")

    if not google_creds_json_str or not gemini_api_key:
        raise KeyError("Secrets not found, falling back to environment variables.")

    # Write temp file for Google Vision client
    with open("gcp-credentials.json", "w") as f:
        f.write(google_creds_json_str)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-credentials.json"
    
    genai.configure(api_key=gemini_api_key)

except (FileNotFoundError, KeyError):
    st.warning("🔑 API secrets not configured for deployment. Relying on local environment variables.", icon="⚠️")
    # Fallback for local development
    if "GEMINI_API_KEY" not in os.environ or "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        st.error("Please set GEMINI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS environment variables for local execution.")
        st.stop()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# --- Core Functions (Backend Logic) ---

@st.cache_data # Caching is GOOD here. OCR result for the same image won't change.
def get_text_from_image(image_content):
    """Uses Google Vision API to perform OCR on the image bytes."""
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_content)
        response = client.text_detection(image=image)
        if response.text_annotations:
            return response.text_annotations[0].description
        return None
    except Exception as e:
        st.error(f"Error calling Google Vision API: {e}")
        return None

# THIS CACHE WAS THE PROBLEM. IT HAS BEEN REMOVED.
# The function will now run every single time it's called.
def analyze_with_gemini(raw_text: str):
    """Uses Gemini API to analyze the raw text and extract ingredients."""
    # Use the model name you confirmed works.
    model = genai.GenerativeModel('gemini-2.5-flash') # Or 'gemini-1.0-pro' if flash has issues
    prompt = """
Extract ingredient lists from the following texts. The ingredient list should start with the first ingredient and end with the last ingredient. It should not include allergy, label or origin information.
The output format must be a single JSON list containing one element per ingredient list. If there are ingredients in several languages, the output JSON list should contain as many elements as detected languages. Each element should have two fields:
- a "text" field containing the detected ingredient list. The text should be a substring of the original text, you must not alter the original text.
- a "lang" field containing the detected language of the ingredient list.
Don't output anything else than the expected JSON list.
    """
    full_prompt = f"{prompt}\n\n{raw_text}"
    try:
        response = model.generate_content(full_prompt)
        # Clean the response to ensure it's valid JSON
        json_response = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(json_response)
    except Exception as e:
        st.error(f"Error during Gemini analysis: {e}")
        return None

# --- Main Application Logic ---
uploaded_file = st.file_uploader(
    "Choose a product image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Get image content as bytes
    image_content = uploaded_file.getvalue()
    
    with st.spinner("Analyzing... This might take a few seconds."):
        # Step 1: Extract raw text using Google Vision
        st.markdown("---")
        st.info("👁️ **Step 1: Extracting Text with Google Vision**")
        raw_text = get_text_from_image(image_content)
        
        if raw_text:
            with st.expander("View Raw Extracted Text"):
                st.text_area("Raw Text", raw_text, height=150)
            
            # Step 2: Analyze text and extract ingredients using Gemini
            st.info("🧠 **Step 2: Analyzing Ingredients with Gemini**")
            structured_ingredients = analyze_with_gemini(raw_text)
            
            if structured_ingredients:
                st.success("🎉 Ingredients extracted successfully!")
                st.json(structured_ingredients)
            else:
                st.error("Gemini could not analyze the text or returned an error.")
        else:
            st.error("No text could be found in the image.")