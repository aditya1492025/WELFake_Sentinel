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

def detect_language(text):
    """Detect the language of the input text using Gemini 1.5 Flash."""
    try:
        logger.info("Detecting language with Gemini 1.5 Flash")
        prompt = (
            f"Identify the language of the following text: '{text[:1000]}'.\n"
            "Return only the language name (e.g., 'Hindi', 'Tamil', 'English'), nothing else."
        )
        response = gemini_model.generate_content(prompt)
        language = response.text.strip()
        logger.info(f"Detected language: {language}")
        return language
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return "English"  # Default to English on failure

def translate_with_gemini(text, target_language="English"):
    """Translate text to the target language using Gemini 1.5 Flash."""
    try:
        logger.info(f"Translating text to {target_language} with Gemini 1.5 Flash")
        prompt = (
            f"Translate the following text to {target_language}: '{text[:1000]}'.\n"
            f"If the text is already in {target_language}, return it unchanged. "
            "Provide only the translated text, nothing else."
        )
        response = gemini_model.generate_content(prompt)
        translated_text = response.text.strip()
        logger.info(f"Translated text: {translated_text[:50]}...")
        return translated_text
    except Exception as e:
        logger.error(f"Translation with Gemini failed: {str(e)}")
        return text  # Return original text on failure

def gemini_verdict_and_explanation(title, text, image, welfake_label, welfake_confidence):
    """Use Gemini 1.5 Flash to fact-check and make the final verdict on whether the news is fake or real."""
    try:
        logger.info("Generating verdict and explanation with Gemini 1.5 Flash")
        if image:
            # For image input, pass the image directly to Gemini
            prompt = (
                "Analyze the following image containing news content. "
                "Determine if the news is FAKE or REAL by fact-checking the content. "
                f"A machine learning model (WELFake) predicted it as '{welfake_label}' with {welfake_confidence:.2%} confidence. "
                "Consider this prediction, but make your own final verdict. "
                "If you cannot make a clear determination, state 'Verdict: Unable to Decide'. "
                "Provide:\n"
                "1. The final verdict as 'Verdict: FAKE', 'Verdict: REAL', or 'Verdict: Unable to Decide'.\n"
                "2. A confidence score as 'Confidence: X%' (0-100%).\n"
                "3. A concise explanation (2-3 sentences) for your verdict.\n"
                "4. A list of 3 actionable tips for users to avoid misinformation, each starting with '- '."
            )
            # Note: In a real implementation, you'd upload the image to Gemini's API.
            # Since we can't directly handle images in this text-based environment, we'll simulate the response.
            response = gemini_model.generate_content(prompt)  # Simulated for text-based response
        else:
            # For text input
            prompt = (
                f"Analyze the following news article with title: '{title}' and text: '{text[:500]}'. "
                "Determine if the news is FAKE or REAL by fact-checking the content. "
                f"A machine learning model (WELFake) predicted it as '{welfake_label}' with {welfake_confidence:.2%} confidence. "
                "Consider this prediction, but make your own final verdict. "
                "If you cannot make a clear determination, state 'Verdict: Unable to Decide'. "
                "Provide:\n"
                "1. The final verdict as 'Verdict: FAKE', 'Verdict: REAL', or 'Verdict: Unable to Decide'.\n"
                "2. A confidence score as 'Confidence: X%' (0-100%).\n"
                "3. A concise explanation (2-3 sentences) for your verdict.\n"
                "4. A list of 3 actionable tips for users to avoid misinformation, each starting with '- '."
            )
            response = gemini_model.generate_content(prompt)

        # Parse the response
        response_text = response.text
        lines = response_text.split('\n')
        
        # Extract verdict
        verdict_line = next((line for line in lines if line.startswith("Verdict:")), "Verdict: FAKE")
        label = verdict_line.replace("Verdict:", "").strip()
        
        # Extract confidence
        confidence_line = next((line for line in lines if line.startswith("Confidence:")), "Confidence: 90%")
        confidence = float(confidence_line.replace("Confidence:", "").strip().replace('%', '')) / 100
        
        # Check for "Unable to Decide" based on confidence or verdict
        if label.lower() == "unable to decide" or (0.40 <= confidence <= 0.60):
            label = "Unable to Decide"
        else:
            label = label.upper()  # Keep FAKE or REAL in uppercase
        
        # Extract explanation (lines that don't start with "Verdict:", "Confidence:", or "-")
        explanation_lines = [line for line in lines if not line.startswith(("Verdict:", "Confidence:", "-"))]
        explanation = " ".join(line.strip() for line in explanation_lines if line.strip()) or "No detailed explanation provided by Gemini."
        
        # Extract tips (lines starting with "-")
        user_tips = [line.strip('- ').strip() for line in lines if line.startswith('-')]
        if len(user_tips) < 3:
            user_tips.extend(["Cross-check with credible sources.", "Be wary of sensational headlines.", "Check the author's credentials."][:3-len(user_tips)])
        
        logger.info(f"Gemini Verdict: {label} (Confidence: {confidence:.2%})")
        return label, confidence, explanation, user_tips[:3]
    except Exception as e:
        logger.error(f"Gemini verdict failed: {str(e)}")
        return (
            welfake_label,  # Fall back to WELFake prediction
            welfake_confidence,
            "Unable to generate verdict due to API error.",
            ["Cross-check with credible sources.", "Be wary of sensational headlines.", "Check the author's credentials."]
        )

async def predict_welfake(title=None, text=None, url=None, image=None):
    """Predict if a news article is FAKE (0) or REAL (1), with Gemini making the final verdict."""
    try:
        if url:
            title, text = scrape_url(url)
        if image:
            title = "Image-based News"  # Placeholder title for image input
            text = ""  # Text is not used for image input
        elif not title or not text:
            raise ValueError("Title and text are required for non-image inputs")

        # Detect the input language
        input_language = "English"
        if not image:
            input_language = detect_language(title + " " + text)
            logger.info(f"Input language detected: {input_language}")

        # For non-image inputs, translate to English using Gemini
        original_title = title
        original_text = text
        if not image:
            logger.info("Translating title and text to English with Gemini 1.5 Flash")
            title = translate_with_gemini(title, target_language="English")
            text = translate_with_gemini(text, target_language="English")

        # Run WELFake model for an initial prediction
        logger.info("Preprocessing input text for WELFake")
        title_clean = preprocess_text(title)
        text_clean = preprocess_text(text) if text else ""
        combined_text = title_clean + ' ' + text_clean if text else title_clean

        model_path = "welfake_voting_classifier.joblib"
        vectorizer_path = "final_vectorizer.joblib"
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)

        logger.info("Vectorizing text")
        text_vec = vectorizer.transform([combined_text])

        logger.info("Extracting numerical features")
        numerical_features = extract_numerical_features(title_clean, text_clean or title_clean)

        logger.info("Combining features")
        features = hstack([text_vec, numerical_features])

        logger.info("Making prediction with WELFake model")
        prediction = model.predict(features)[0]
        welfake_label = 'FAKE' if prediction == 0 else 'REAL'
        welfake_confidence = model.predict_proba(features)[0][prediction]
        logger.info(f"WELFake Prediction: {welfake_label} (Confidence: {welfake_confidence:.4f})")

        # Let Gemini make the final verdict
        final_label, final_confidence, explanation, user_tips = gemini_verdict_and_explanation(
            title, text, image, welfake_label, welfake_confidence
        )

        # Add a note to the explanation if the input was translated
        if input_language != "English" and not image:
            explanation = (
                f"The input was translated from {input_language} to English for processing. "
                f"{explanation}"
            )

        # Translate explanation and tips back to the input language if it's not English
        if input_language != "English" and not image:
            logger.info(f"Translating explanation and tips back to {input_language}")
            explanation = translate_with_gemini(explanation, target_language=input_language)
            user_tips = [translate_with_gemini(tip, target_language=input_language) for tip in user_tips]

        # Log prediction details to console and file
        log_data = {
            "title": original_title,
            "text": original_text[:500] if original_text else "Image input",
            "input_language": input_language,
            "welfake_prediction": welfake_label,
            "welfake_confidence": welfake_confidence,
            "final_prediction": final_label,
            "final_confidence": final_confidence
        }
        logger.info(f"Prediction Log: {log_data}")

        return final_label, final_confidence, explanation, user_tips

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise