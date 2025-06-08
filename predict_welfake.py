# This script contains the prediction logic for the fake news detector, integrated with Streamlit
import pandas as pd
import re
import joblib
from scipy.sparse import hstack
import logging
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configure Gemini 1.5 Flash (load from environment variable)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not set. Set the 'GEMINI_API_KEY' environment variable.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def preprocess_text(text):
    """Preprocess text by converting to lowercase and removing special characters."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_numerical_features(title, text):
    """Extract numerical features to match training data (simplified)."""
    numerical = {
        'polarity': 0.0,
        'subjectivity': 0.0,
        'num_words': len(text.split()),
        'num_chars': len(text),
        'avg_word_len': sum(len(word) for word in text.split()) / (len(text.split()) or 1),
        'num_unique_words': len(set(text.split())),
        'num_stop_words': 0.0,
        'num_punctuations': 0.0,
        'num_uppercase': sum(1 for c in text if c.isupper())
    }
    return pd.DataFrame([numerical])

def scrape_url(url):
    """Scrape title and text from a URL."""
    try:
        logger.info(f"Scraping URL: {url}")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No Title"
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs) if paragraphs else "No Content"
        logger.info(f"Scraped Title: {title[:50]}..., Text: {text[:50]}...")
        return title, text
    except Exception as e:
        logger.error(f"URL scraping failed: {str(e)}")
        return "Sample Title", "Sample text for testing purposes."

def translate_with_gemini(text):
    """Translate text to English using Gemini 1.5 Flash."""
    try:
        logger.info("Translating text to English with Gemini 1.5 Flash")
        prompt = (
            f"Translate the following text to English: '{text[:1000]}'.\n"
            "If the text is already in English, return it unchanged. "
            "Provide only the translated text, nothing else."
        )
        response = gemini_model.generate_content(prompt)
        translated_text = response.text.strip()
        logger.info(f"Translated text: {translated_text[:50]}...")
        return translated_text
    except Exception as e:
        logger.error(f"Translation with Gemini failed: {str(e)}")
        return text  # Return original text on failure

def gemini_explanation(title, text, predicted_label):
    """Use Gemini 1.5 Flash to explain the prediction and provide user tips."""
    try:
        logger.info("Generating explanation with Gemini 1.5 Flash")
        prompt = (
            f"Given the news article with title: '{title}' and text: '{text[:500]}', "
            f"the model predicted it as '{predicted_label}'. "
            "Provide a concise explanation (2-3 sentences) for why this prediction makes sense. "
            "Then, list 3 actionable tips for users to improve their media literacy and avoid misinformation."
        )
        response = gemini_model.generate_content(prompt)
        explanation = response.text
        explanation_lines = explanation.split('\n')
        explanation_text = '\n'.join(line for line in explanation_lines if not line.startswith('-'))
        tips = [line.strip('- ').strip() for line in explanation_lines if line.startswith('-')]
        if len(tips) < 3:
            tips.extend(["Cross-check with credible sources.", "Be wary of sensational headlines.", "Check the author's credentials."][:3-len(tips)])
        logger.info(f"Gemini Explanation: {explanation_text[:100]}...")
        return explanation_text, tips[:3]
    except Exception as e:
        logger.error(f"Gemini explanation failed: {str(e)}")
        return (
            "Unable to generate explanation due to API error.",
            ["Cross-check with credible sources.", "Be wary of sensational headlines.", "Check the author's credentials."]
        )

async def predict_welfake(title=None, text=None, url=None):
    """Predict if a news article is FAKE (0) or REAL (1), with Gemini for translation and explanation."""
    try:
        if url:
            title, text = scrape_url(url)
        if not title or not text:
            raise ValueError("Title and text are required")

        # Use Gemini 1.5 Flash to translate title and text to English
        logger.info("Translating title and text with Gemini 1.5 Flash")
        title = translate_with_gemini(title)
        text = translate_with_gemini(text)

        logger.info("Preprocessing input text")
        title_clean = preprocess_text(title)
        text_clean = preprocess_text(text)
        combined_text = title_clean + ' ' + text_clean

        # Adjusted paths for Streamlit deployment
        model_path = "welfake_voting_classifier.joblib"
        vectorizer_path = "final_vectorizer.joblib"
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)

        logger.info("Vectorizing text")
        text_vec = vectorizer.transform([combined_text])

        logger.info("Extracting numerical features")
        numerical_features = extract_numerical_features(title_clean, text_clean)

        logger.info("Combining features")
        features = hstack([text_vec, numerical_features])

        logger.info("Making prediction with WELFake model")
        prediction = model.predict(features)[0]
        label = 'FAKE' if prediction == 0 else 'REAL'
        confidence = model.predict_proba(features)[0][prediction]
        logger.info(f"WELFake Prediction: {label} (Confidence: {confidence:.4f})")

        explanation, user_tips = gemini_explanation(title, text, label)

        return label, confidence, explanation, user_tips

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise