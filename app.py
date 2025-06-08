# Streamlit frontend for fake news detection portal with Bootstrap styling
import streamlit as st
import asyncio
from predict_welfake import predict_welfake

# Streamlit page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Add Bootstrap CDN for styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stTextInput, .stTextArea {
            margin-bottom: 1rem;
        }
        .result-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .prediction-fake {
            color: #dc3545;
            font-weight: bold;
        }
        .prediction-real {
            color: #28a745;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 class='text-center mb-4'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='text-center text-muted mb-4'>Enter a news article's title and text, or a URL, to check if it's fake or real.</p>", unsafe_allow_html=True)

# Input form
with st.form(key="news_form", clear_on_submit=False):
    input_type = st.radio("Choose input type:", ("Title and Text", "URL"), horizontal=True)
    
    if input_type == "Title and Text":
        title = st.text_input("Title", placeholder="Enter the news title")
        text = st.text_area("Text", placeholder="Enter the news text", height=150)
        url = None
    else:
        url = st.text_input("URL", placeholder="Enter the news URL (e.g., https://www.bbc.com/news)")
        title = None
        text = None

    submit_button = st.form_submit_button("Check News", use_container_width=True, type="primary")

# Process form submission
if submit_button:
    if input_type == "Title and Text" and (not title or not text):
        st.warning("Please provide both a title and text.")
    elif input_type == "URL" and not url:
        st.warning("Please provide a URL.")
    else:
        with st.spinner("Analyzing news..."):
            try:
                # Run the async prediction function
                label, confidence, explanation, user_tips = asyncio.run(predict_welfake(title=title, text=text, url=url))
                
                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                prediction_class = "prediction-fake" if label == "FAKE" else "prediction-real"
                st.markdown(f"<h3>Prediction: <span class='{prediction_class}'>{label}</span></h3>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Confidence:</strong> {confidence:.2%}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Explanation:</strong> {explanation}</p>", unsafe_allow_html=True)
                st.markdown("<p><strong>Tips to Avoid Misinformation:</strong></p>", unsafe_allow_html=True)
                for tip in user_tips:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='text-center text-muted'>Built for INNOHACK 2025 by Team UNIT AI</p>", unsafe_allow_html=True)