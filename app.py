import io
import os
import json
import streamlit as st
from google.cloud import vision
import google.generativeai as genai

# --- إعداد واجهة المستخدم ---
st.set_page_config(page_title=" Mukawwin | مُكوِّن", layout="centered")
st.title(" Mukawwin | مُكوِّن 🤖")
st.write("حلل مكونات أي منتج غذائي عبر التقاط صورة أو رفعها مباشرة.")

# --- إعداد واجهات برمجة التطبيقات (APIs) ---
# هذا الجزء يبقى كما هو، مصمم للعمل محليًا ومع النشر
try:
    google_creds_json_str = st.secrets["gcp_service_account_json_str"]
    gemini_api_key = st.secrets["gemini_api_key"]
    with open("gcp-credentials.json", "w") as f:
        f.write(google_creds_json_str)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-credentials.json"
    genai.configure(api_key=gemini_api_key)
except (FileNotFoundError, KeyError):
    st.warning("🔑 يتم الآن الاعتماد على متغيرات البيئة المحلية للتشغيل.", icon="⚠️")
    if "GEMINI_API_KEY" not in os.environ or "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        st.error("يرجى إعداد متغيرات البيئة GEMINI_API_KEY و GOOGLE_APPLICATION_CREDENTIALS للتشغيل المحلي.")
        st.stop()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# --- الدوال الأساسية (المنطق الخلفي) ---
@st.cache_data
def get_text_from_image(image_content):
    """يستخدم Google Vision API لاستخراج النص الخام من الصورة."""
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_content)
        response = client.text_detection(image=image)
        return response.text_annotations[0].description if response.text_annotations else None
    except Exception as e:
        st.error(f"حدث خطأ أثناء الاتصال بـ Google Vision: {e}")
        return None

# تمت إزالة التخزين المؤقت من هنا لإصلاح مشكلة الردود القديمة
def analyze_with_gemini(raw_text: str):
    """يستخدم Gemini API لتحليل النص الخام واستخراج المكونات."""
    model = genai.GenerativeModel('gemini-2.5-flash')
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
        json_response = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(json_response)
    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل بواسطة Gemini: {e}")
        return None

# --- NEW: Create tabs for input options ---
tab1, tab2 = st.tabs(["📷 التقاط صورة", "⬆️ رفع ملف"])

with tab1:
    camera_photo = st.camera_input(
        "التقط صورة لملصق المكونات", key="camera_uploader"
    )

with tab2:
    uploaded_file = st.file_uploader(
        "أو اختر صورة من جهازك", type=["jpg", "jpeg", "png"], key="file_uploader"
    )

# --- منطق تشغيل التطبيق (موحد لكلا الخيارين) ---
image_source = camera_photo or uploaded_file

if image_source is not None:
    st.image(image_source, caption="الصورة المحددة", use_column_width=True)
    
    image_content = image_source.getvalue()
    
    with st.spinner("🧠 جاري التحليل... قد يستغرق الأمر بضع ثوانٍ"):
        st.markdown("---")
        st.info("👁️ **الخطوة 1: استخراج النص باستخدام Google Vision**")
        raw_text = get_text_from_image(image_content)
        
        if raw_text:
            with st.expander("عرض النص الخام المستخرج"):
                st.text_area("النص الخام", raw_text, height=150)
            
            st.info("🤖 **الخطوة 2: تحليل المكونات باستخدام Gemini**")
            structured_ingredients = analyze_with_gemini(raw_text)
            
            if structured_ingredients:
                st.success("🎉 تم استخراج المكونات بنجاح!")
                st.json(structured_ingredients)
            else:
                st.error("لم يتمكن Gemini من تحليل النص.")
        else:
            st.error("لم يتم العثور على أي نص في الصورة.")