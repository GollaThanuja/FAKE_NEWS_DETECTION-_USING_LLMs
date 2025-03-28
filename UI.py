import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Set Streamlit Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Apply Custom CSS for Centering and Styling
# Apply Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        text-align: center;
    }
    .title {
        margin-top: 500px;  /* Adjust this value to move it up */
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        max-width: 700px;
        margin: auto;
        max-height: 750px;
    }
    h1 {
        color: #f8f9fa;
        font-size: 32px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Center the radio buttons */
    .stRadio > label {
        display: flex;
        justify-content: center;
        font-size: 22px;
        font-weight: bold;
        color: #f8f9fa;
    }
    /* Align radio options horizontally */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: -40px;
    }
    </style>
""", unsafe_allow_html=True)



# Load the ResNet-50 Model for Images
image_model_path = "resnet50_fake_news.pth"
image_model = models.resnet50(pretrained=False)  
num_ftrs = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_ftrs, 2)  
image_model.load_state_dict(torch.load(image_model_path, map_location=torch.device('cpu')))
image_model.eval()

# Load the XLM-RoBERTa Model for Text
text_model_path = "xlm-roberta-base"
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path)
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)

# Define Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Predict Image Class with Confidence Score
def predict_image(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = image_model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    return prediction.item(), confidence.item()

# Predict Text Class with Confidence Score
def predict_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = text_model(**inputs).logits
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    return prediction.item(), confidence.item()

# Streamlit UI
  # Adds space above
st.markdown("<h1 style='text-align: center; color: black;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write("🔎 Upload an **image** or enter **text** to verify its authenticity.")

# Centered Options
st.markdown("<center><h4 style='text-align: center;'>Select Input Type:</h4></center>", unsafe_allow_html=True)
option = st.radio("", ("📝 Text", "🖼️ Image"), horizontal=True)

if option == "📝 Text":
    user_text = st.text_area("📝 Enter news text here:")
    if st.button("🔍 Analyze Text"):
        if user_text.strip():
            prediction, confidence = predict_text(user_text)
            label = "🟩 Real News" if prediction == 0 else "🟥 Fake News"
            
            st.markdown(f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True)
            
            # Confidence Bar with Percentage
            st.progress(int(confidence * 100))
            st.write(f"Confidence: **{confidence * 100:.2f}%**")
            
            # Provide Explanation
            if prediction == 1:
                st.success("✅ The text appears **authentic** based on the model's analysis.")
            else:
                st.warning("⚠️ The text might be **fake news**. Cross-check with trusted sources.")

elif option == "🖼️ Image":
    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Arrange Layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Analyze Image"):
                prediction, confidence = predict_image(image)
                label = "🟩 Real News" if prediction == 1 else "🟥 Fake News"
                
                st.markdown(f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True)
                
                # Show Confidence Bar
                st.progress(int(confidence * 100))
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
                
                # Provide Explanation
                if prediction == 1:
                    st.success("✅ The image appears to be **authentic**.")
                else:
                    st.warning("⚠️ This image might be **misleading**. Verify with trusted sources.")
