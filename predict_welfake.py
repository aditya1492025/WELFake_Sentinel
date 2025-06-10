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
from dateutil.parser import parse
import datetime
import base64
from io import BytesIO

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

# Configure Google Custom Search Engine (CSE)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Google API key or CSE ID not set. Set 'GOOGLE_API_KEY' and 'GOOGLE_CSE_ID' environment variables.")

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

def extract_dates(text):
    """Extract dates from the text using dateutil and regex."""
    try:
        # Replace "today" with the current date
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        text = text.replace("today", current_date, 1)

        # Regex to find date patterns (e.g., June 10, 2025, 10/06/2025)
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
        dates = re.findall(date_pattern, text)

        # Parse dates using dateutil
        parsed_dates = []
        for date in dates:
            try:
                parsed_date = parse(date, fuzzy=True)
                parsed_dates.append(parsed_date.strftime("%B %d, %Y"))
            except ValueError:
                continue

        return parsed_dates if parsed_dates else [current_date]  # Default to current date if none found
    except Exception as e:
        logger.error(f"Date extraction failed: {str(e)}")
        return [datetime.datetime.now().strftime("%B %d, %Y")]

def search_with_google_cse(query):
    """Search for news using Google Custom Search Engine API."""
    try:
        logger.info(f"Searching Google CSE with query: {query}")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": 5,  # Fetch top 5 results
            "sort": "date",  # Sort by date for latest news
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("items", [])
        return results
    except Exception as e:
        logger.error(f"Google CSE search failed: {str(e)}")
        return []

def scrape_search_results(results):
    """Scrape titles, snippets, and publication dates from Google CSE results."""
    scraped_data = []
    for item in results:
        try:
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "No Snippet")
            link = item.get("link", "")
            # Attempt to fetch publication date from the page
            response = requests.get(link, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for common date meta tags or elements
            date = None
            meta_date = soup.find("meta", {"name": "date"}) or soup.find("meta", {"property": "article:published_time"})
            if meta_date and meta_date.get("content"):
                date = parse(meta_date.get("content")).strftime("%B %d, %Y")
            if not date:
                # Fallback to extract from snippet
                date_matches = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', snippet)
                date = date_matches[0] if date_matches else "Unknown Date"
            scraped_data.append({"title": title, "snippet": snippet, "date": date, "link": link})
        except Exception as e:
            logger.error(f"Scraping search result failed: {str(e)}")
            continue
    return scraped_data

def fact_check_news_with_gemini(title, text, extracted_dates):
    """Use Gemini to fact-check the entire news content with scraped data (RAG mechanism)."""
    try:
        logger.info("Fact-checking news content with Gemini 1.5 Flash")
        # Create a search query for the entire news claim
        search_query = f"{title} {text[:100]} {' '.join(extracted_dates)}"
        search_results = search_with_google_cse(search_query)
        scraped_data = scrape_search_results(search_results)

        # Summarize the scraped data
        scraped_summary = "\n".join(
            f"Source: {data['title']} (Published: {data['date']})\nSnippet: {data['snippet']}"
            for data in scraped_data
        ) if scraped_data else "No relevant search results found."

        prompt = (
            f"Fact-check the following news article:\n"
            f"Title: '{title}'\n"
            f"Text: '{text[:500]}'\n"
            f"Claimed Dates: {', '.join(extracted_dates) if extracted_dates else 'None specified'}\n\n"
            f"Relevant search results:\n{scraped_summary}\n\n"
            "Determine if the news article is accurate based on the search results. "
            "Consider both the content of the claim and the accuracy of any dates mentioned. "
            "Provide:\n"
            "1. Verdict as 'News: Accurate', 'News: Inaccurate', or 'News: Unable to Verify'.\n"
            "2. A confidence score as 'Confidence: X%' (0-100%).\n"
            "3. A concise explanation (2-3 sentences) for your verdict, addressing both the content and dates."
        )
        response = gemini_model.generate_content(prompt)
        response_text = response.text
        lines = response_text.split('\n')

        # Extract verdict
        verdict_line = next((line for line in lines if line.startswith("News:")), "News: Unable to Verify")
        news_verdict = verdict_line.replace("News:", "").strip()

        # Extract confidence
        confidence_line = next((line for line in lines if line.startswith("Confidence:")), "Confidence: 50%")
        news_confidence = float(confidence_line.replace("Confidence:", "").strip().replace('%', '')) / 100

        # Extract explanation
        explanation_lines = [line for line in lines if not line.startswith(("News:", "Confidence:"))]
        news_explanation = " ".join(line.strip() for line in explanation_lines if line.strip()) or "No detailed explanation provided."

        logger.info(f"News Fact-Check Verdict: {news_verdict} (Confidence: {news_confidence:.2%})")
        return news_verdict, news_confidence, news_explanation
    except Exception as e:
        logger.error(f"News fact-checking with Gemini failed: {str(e)}")
        return "News: Unable to Verify", 0.5, "Unable to verify news due to API error."

def gemini_verdict_and_explanation(title, text, image, welfake_label=None, welfake_confidence=None, news_verdict=None, news_confidence=None, news_explanation=None):
    """Use Gemini 1.5 Flash to generate a verdict, confidence, explanation, and tips."""
    try:
        logger.info("Generating verdict and explanation with Gemini 1.5 Flash")
        news_info = ""
        if news_verdict:
            news_info = (
                f"News Fact-Check Result (via RAG):\n"
                f"- Verdict: {news_verdict}\n"
                f"- Confidence: {news_confidence:.2%}\n"
                f"- Explanation: {news_explanation}\n"
            )

        welfake_info = ""
        if welfake_label and welfake_confidence is not None:
            welfake_info = (
                f"Supplementary WELFake Model Prediction:\n"
                f"- Prediction: {welfake_label}\n"
                f"- Confidence: {welfake_confidence:.2%}\n"
            )

        if image:
            # Read and encode the image for Gemini API
            logger.info("Processing image input for Gemini fact-checking")
            image_bytes = image.read()  # Read the image as bytes
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')  # Encode to base64 string

            # Create the prompt and pass the image to Gemini
            prompt = (
                "Analyze the following image containing news content. "
                "Fact-check the content and provide:\n"
                "1. Verdict as 'Gemini Verdict: Accurate', 'Gemini Verdict: Inaccurate', or 'Gemini Verdict: Unable to Verify'.\n"
                "2. A confidence score as 'Confidence: X%' (0-100%).\n"
                "3. A concise explanation (2-3 sentences) for your verdict.\n"
                "4. A list of 3 actionable tips for users to avoid misinformation, each starting with '- '."
            )
            # Pass the image and prompt to Gemini
            response = gemini_model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": encoded_image}]
            )
        else:
            # For text input
            prompt = (
                f"Analyze the following news article with title: '{title}' and text: '{text[:500]}'. "
                f"{news_info}"
                f"{welfake_info}"
                "Fact-check the news content independently, considering the RAG fact-check results and WELFake prediction as references. "
                "Provide:\n"
                "1. Verdict as 'Gemini Verdict: Accurate', 'Gemini Verdict: Inaccurate', or 'Gemini Verdict: Unable to Verify'.\n"
                "2. A confidence score as 'Confidence: X%' (0-100%).\n"
                "3. A concise explanation (2-3 sentences) for your verdict, based on all available information.\n"
                "4. A list of 3 actionable tips for users to avoid misinformation, each starting with '- '."
            )
            response = gemini_model.generate_content(prompt)

        # Parse the response
        response_text = response.text
        lines = response_text.split('\n')
        
        # Extract verdict
        verdict_line = next((line for line in lines if line.startswith("Gemini Verdict:")), "Gemini Verdict: Unable to Verify")
        gemini_verdict = verdict_line.replace("Gemini Verdict:", "").strip()

        # Extract confidence
        confidence_line = next((line for line in lines if line.startswith("Confidence:")), "Confidence: 50%")
        gemini_confidence = float(confidence_line.replace("Confidence:", "").strip().replace('%', '')) / 100

        # Extract explanation (lines that don't start with "-" or other fields)
        explanation_lines = [line for line in lines if not line.startswith(("- ", "Gemini Verdict:", "Confidence:"))]
        explanation = " ".join(line.strip() for line in explanation_lines if line.strip()) or "No detailed explanation provided by Gemini."
        
        # Extract tips (lines starting with "-")
        user_tips = [line.strip('- ').strip() for line in lines if line.startswith('-')]
        if len(user_tips) < 3:
            user_tips.extend(["Cross-check with credible sources.", "Be wary of sensational headlines.", "Check the author's credentials."][:3-len(user_tips)])
        
        return gemini_verdict, gemini_confidence, explanation, user_tips[:3]
    except Exception as e:
        logger.error(f"Gemini verdict and explanation failed: {str(e)}")
        return (
            "Gemini Verdict: Unable to Verify",
            0.5,
            "Unable to generate explanation due to API error.",
            ["Cross-check with credible sources.", "Be wary of sensational headlines.", "Check the author's credentials."]
        )

def predict_welfake(title=None, text=None, url=None, image=None):
    """Predict if a news article is FAKE (0) or REAL (1), with the final verdict based on both RAG and Gemini."""
    try:
        if url:
            title, text = scrape_url(url)
        if image:
            title = "Image-based News"  # Placeholder title for image input
            text = ""  # Text is not used for image input
            logger.info("Image input detected, proceeding with Gemini fact-checking")
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

        # Fact-check the entire news content using Google CSE and Gemini (RAG mechanism)
        news_verdict, news_confidence, news_explanation = None, None, None
        if not image:
            # Extract dates from the text for better search queries
            combined_text_for_dates = f"{title} {text}"
            extracted_dates = extract_dates(combined_text_for_dates)
            logger.info(f"Extracted dates: {extracted_dates}")

            # Fact-check the news content with Gemini (RAG)
            news_verdict, news_confidence, news_explanation = fact_check_news_with_gemini(
                title, text, extracted_dates
            )

        # Run WELFake model to provide supplementary context
        welfake_label, welfake_confidence = None, None
        if not image:
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

        # Get Gemini's independent verdict, confidence, explanation, and tips
        gemini_verdict, gemini_confidence, explanation, user_tips = gemini_verdict_and_explanation(
            title, text, image, welfake_label, welfake_confidence,
            news_verdict, news_confidence, news_explanation
        )

        # Combine RAG and Gemini verdicts
        # Simplify verdicts to a common format for comparison
        rag_verdict_simple = news_verdict.replace("News: ", "") if news_verdict else "Unable to Verify"
        gemini_verdict_simple = gemini_verdict.replace("Gemini Verdict: ", "")

        # Determine combined verdict
        if rag_verdict_simple == gemini_verdict_simple:
            combined_verdict = rag_verdict_simple  # Both agree
        else:
            combined_verdict = "Unable to Verify"  # Disagree, so "Unable to Decide"

        # Determine combined confidence
        if combined_verdict == "Unable to Verify":
            combined_confidence = min(news_confidence or 0.5, gemini_confidence)  # Lower confidence if disagreement
        else:
            combined_confidence = ((news_confidence or 0.5) + gemini_confidence) / 2  # Average confidence if agreement

        # Map combined verdict to final prediction
        if combined_verdict == "Accurate":
            final_label = "REAL"
        elif combined_verdict == "Inaccurate":
            final_label = "FAKE"
        else:  # "Unable to Verify"
            final_label = "Unable to Decide"

        # Check for "Unable to Decide" based on confidence
        if 0.40 <= combined_confidence <= 0.60:
            final_label = "Unable to Decide"

        final_confidence = combined_confidence

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
            "news_verdict": news_verdict,
            "news_confidence": news_confidence,
            "gemini_verdict": gemini_verdict,
            "gemini_confidence": gemini_confidence,
            "final_prediction": final_label,
            "final_confidence": final_confidence
        }
        logger.info(f"Prediction Log: {log_data}")

        return final_label, final_confidence, explanation, user_tips

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise