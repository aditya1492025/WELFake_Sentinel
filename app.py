# Streamlit frontend for fake news detection portal with Bootstrap styling
import streamlit as st
import asyncio
import nest_asyncio
from predict_welfake import predict_welfake

# Apply nest_asyncio to allow asyncio.run in Streamlit's event loop
nest_asyncio.apply()

# Streamlit page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Add Bootstrap CDN and custom CSS for styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .custom-container {
            background-color: #f8f9fa;
            padding: 2px;
            min-height: 1vh;
        }
        .input-field {
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
        .prediction-uncertain {
            color: #ffc107;
            font-weight: bold;
        }
        .form-container {
            background-color: #ffffff;
            padding: 2px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='text-center mb-4'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='text-center text-muted mb-4'>Enter a news article's title and text, a URL, or upload an image to check if it's fake or real.</p>", unsafe_allow_html=True)

# Input form
with st.form(key="news_form", clear_on_submit=False):
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    input_type = st.radio("Choose input type:", ("Title and Text", "URL", "Image"), horizontal=True, key="input_type")
    
    if input_type == "Title and Text":
        title = st.text_input("Title", placeholder="Enter the news title", key="title_input")
        text = st.text_area("Text", placeholder="Enter the news text", height=150, key="text_input")
        url = None
        image = None
    elif input_type == "URL":
        url = st.text_input("URL", placeholder="Enter the news URL (e.g., https://www.bbc.com/news)", key="url_input")
        title = None
        text = None
        image = None
    else:  # Image
        image = st.file_uploader("Upload an image containing news content", type=["jpg", "jpeg", "png"], key="image_input")
        title = None
        text = None
        url = None

    submit_button = st.form_submit_button("Check News", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# Process form submission
if submit_button:
    if input_type == "Title and Text" and (not title or not text):
        st.warning("Please provide both a title and text.")
    elif input_type == "URL" and not url:
        st.warning("Please provide a URL.")
    elif input_type == "Image" and not image:
        st.warning("Please upload an image.")
    else:
        with st.spinner("Analyzing news..."):
            try:
                # For image input, pass the raw file object (predict_welfake will handle it)
                image_input = image if input_type == "Image" else None

                # Run the async prediction function
                loop = asyncio.get_event_loop()
                label, confidence, explanation, user_tips = loop.run_until_complete(predict_welfake(title=title, text=text, url=url, image=image_input))
                
                # Display success message
                st.success("Analysis complete!")

                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                if label == "Unable to Decide":
                    prediction_class = "prediction-uncertain"
                elif label == "FAKE":
                    prediction_class = "prediction-fake"
                elif label == "REAL":
                    prediction_class = "prediction-real"
                else:
                    prediction_class = "prediction-uncertain" 
                
                # Display prediction and confidence (corrected formatting)
                #st.markdown(f"<h3>Prediction: <span class='{prediction_class}'>{label}</span></h3>", unsafe_allow_html=True)
                #st.markdown(f"<p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Explanation:</strong> {explanation}</p>", unsafe_allow_html=True)
                st.markdown("<p><strong>Tips to Avoid Misinformation:</strong></p>", unsafe_allow_html=True)
                st.markdown("<ul>", unsafe_allow_html=True)
                for tip in user_tips:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                st.markdown("</ul>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='text-center text-muted'>Built for INNOHACK 2025 by Team UNIT AI</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)