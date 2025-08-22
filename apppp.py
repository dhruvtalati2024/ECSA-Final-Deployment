import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import nltk
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import traceback
from wordcloud import WordCloud, STOPWORDS

# --- BRUTE-FORCE NLTK DOWNLOAD ---
# This is the most direct method. It will run every single time the app starts.
# This is the only method that has a chance of working in the stubborn environment.
nltk.download('punkt' )
nltk.download('vader_lexicon')

# --- Configuration and Setup ---
st.set_page_config(page_title="ECSA Tool", layout="wide")

# --- API KEY ---
# Get the API key from Streamlit secrets. This is the most reliable way.
API_KEY = st.secrets.get("API_KEY")

# --- ALL FUNCTIONS IN ONE FILE ---

@st.cache_resource
def load_models_and_data():
    """Loads all necessary models and data files once."""
    try:
        lm_dict = pd.read_csv("LMMD.csv")
        lm_positive_words = set(lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower())
        lm_negative_words = set(lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower())
        lm_uncertainty_words = set(lm_dict[lm_dict['Uncertainty'] > 0]['Word'].str.lower())
        sentiment_analyzer = SentimentIntensityAnalyzer()
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return {
            "vader": sentiment_analyzer, "finbert": finbert,
            "lm": {"positive": lm_positive_words, "negative": lm_negative_words, "uncertainty": lm_uncertainty_words}
        }
    except Exception as e:
        st.error(f"Error loading models or LMMD.csv: {e}")
        return None

def handle_api_request(url, payload, headers):
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def clean_text_api(transcript, api_key):
    if not api_key:
        st.warning("API key not found in secrets. Skipping text cleansing.")
        return transcript
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        "Please remove all operator instructions, legal disclaimers, metadata, and introductory pleasantries "
        "from the following earnings call transcript. Return only the core content, which includes the "
        "prepared remarks from executives and the question-and-answer (Q&A ) session. The output should be clean, "
        f"continuous text.\n\n---\n\n{transcript}"
    )
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}]}
    data = handle_api_request(url, payload, headers)
    if data and data.get("choices"):
        return data["choices"][0]["message"]["content"]
    st.error("Failed to clean text using API. Using original transcript.")
    return transcript

def analyze_sentiment(cleaned_text, models_dict):
    sentences = nltk.sent_tokenize(cleaned_text)
    words = nltk.word_tokenize(cleaned_text.lower())
    finbert_scores = models_dict["finbert"](sentences)
    finbert_pos = sum(1 for s in finbert_scores if s['label'] == 'positive')
    finbert_neg = sum(1 for s in finbert_scores if s['label'] == 'negative')
    finbert_neu = len(sentences) - finbert_pos - finbert_neg
    finbert_avg = (finbert_pos - finbert_neg) / len(sentences) if sentences else 0
    vader_scores = [models_dict["vader"].polarity_scores(s)['compound'] for s in sentences]
    vader_pos = sum(1 for s in vader_scores if s > 0.05)
    vader_neg = sum(1 for s in vader_scores if s < -0.05)
    vader_neu = len(vader_scores) - vader_pos - vader_neg
    vader_avg = sum(vader_scores) / len(vader_scores) if vader_scores else 0
    lm_pos = sum(1 for w in words if w in models_dict["lm"]["positive"])
    lm_neg = sum(1 for w in words if w in models_dict["lm"]["negative"])
    lm_unc = sum(1 for w in words if w in models_dict["lm"]["uncertainty"])
    lm_total_sentiment = lm_pos + lm_neg
    lm_score = (lm_pos - lm_neg) / lm_total_sentiment if lm_total_sentiment > 0 else 0
    return {
        'FinBERT': {'score': finbert_avg, 'counts': {'positive': finbert_pos, 'negative': finbert_neg, 'neutral': finbert_neu}},
        'VADER': {'score': vader_avg, 'counts': {'positive': vader_pos, 'negative': vader_neg, 'neutral': vader_neu}},
        'LM': {'score': lm_score, 'counts': {'positive': lm_pos, 'negative': lm_neg, 'uncertainty': lm_unc}}
    }

def get_market_performance(ticker, call_date):
    try:
        stock = yf.Ticker(ticker)
        start_date = datetime.combine(call_date, datetime.min.time())
        end_date = start_date + timedelta(days=8)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if len(hist) < 2: return 0.0, None
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        return ((end_price - start_price) / start_price) * 100, hist
    except Exception as e:
        st.error(f"Failed to fetch market data for {ticker}: {e}")
        return 0.0, None

def create_visualizations(sentiment_results, market_history, ticker, cleaned_text):
    figs = {}
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    methods = ['FinBERT', 'VADER', 'LM']
    scores = [sentiment_results[m]['score'] for m in methods]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    ax1.bar(methods, scores, color=colors)
    ax1.set_title('Normalized Sentiment Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sentiment Score (-1 to 1)')
    ax1.axhline(0, color='grey', linewidth=0.8)
    plt.tight_layout()
    figs['score_comparison'] = fig1
    if market_history is not None and not market_history.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        market_history['Close'].plot(ax=ax2, marker='o', linestyle='-')
        ax2.set_title(f'{ticker} Stock Price: 7-Day Post-Call Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Closing Price (USD)')
        ax2.set_xlabel('Date')
        plt.tight_layout()
        figs['market_performance'] = fig2
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["company", "quarter", "earnings", "call", "revenue", "year", "billion", "million", "financial", "results", "conference", "operator", "question", "analyst", "thank", "thanks", "please", "good", "morning", "afternoon"])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords, collocations=False).generate(cleaned_text)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.set_title('Key Topics Word Cloud', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.tight_layout()
    figs['word_cloud'] = fig3
    return figs

def generate_report_api(cleaned_text, sentiment_results, market_change, api_key):
    if not api_key: return "Report generation failed: No API key."
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"""
    **Objective:** Generate a comprehensive, professional financial report based on the provided earnings call transcript analysis.
    **Report Structure:**
    1.  **Executive Summary:** A brief, high-level overview of the earnings call's key themes, overall sentiment, and subsequent market reaction.
    2.  **Key Topics Discussed:** Identify and summarize the 3-5 most critical topics from the call.
    3.  **Sentiment Analysis Deep Dive:**
        *   **Overall Tone:** Describe the general sentiment of the call.
        *   **Methodology Explanation:** Briefly explain FinBERT, VADER, and the Loughran-McDonald lexicon.
        *   **Results Interpretation:** Analyze the provided sentiment scores: FinBERT Score: {sentiment_results['FinBERT']['score']:.3f}, VADER Score: {sentiment_results['VADER']['score']:.3f}, LM Score: {sentiment_results['LM']['score']:.3f}. Discuss any convergence or divergence.
    4.  **Market Reaction Analysis:**
        *   The stock's price changed by **{market_change:.2f}%** in the 7 days following the call.
        *   Interpret this movement in the context of the sentiment scores.
    5.  **Conclusion:** A concluding paragraph summarizing the findings.
    **Source Transcript (first 1000 characters for context ):**
    ---
    {cleaned_text[:1000]}...
    ---
    Please generate the full report based on this structure and data.
    """
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}]}
    data = handle_api_request(url, payload, headers)
    if data and data.get("choices"): return data["choices"][0]["message"]["content"]
    return "Report generation failed."

def create_pdf_report(report_text, figs):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Earnings Call Sentiment Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))
    report_lines = report_text.split('\n')
    for line in report_lines:
        line = line.strip()
        if line.startswith('**') and line.endswith('**'):
            story.append(Paragraph(line.replace('**', ''), styles['h2']))
        elif line:
            story.append(Paragraph(line, styles['BodyText']))
        story.append(Spacer(1, 0.05 * inch))
    if figs:
        story.append(PageBreak())
        story.append(Paragraph("Visualizations", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))
        fig_order = ['word_cloud', 'score_comparison', 'market_performance']
        for fig_name in fig_order:
            if fig_name in figs:
                img_buffer = BytesIO()
                figs[fig_name].savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img = Image(img_buffer, width=6*inch, height=4*inch, kind='proportional')
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Main App Logic ---
models = load_models_and_data()
st.title("📈 Earnings Call Sentiment Analyzer (ECSA)")
st.markdown("Upload an earnings call transcript, provide the company ticker and call date, and get a full sentiment and market analysis report.")
if not API_KEY: st.warning("API key not found in Streamlit secrets. AI features will be skipped.")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Earnings Call Transcript (.txt)", type="txt")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")
    call_date = st.date_input("Enter Earnings Call Date", datetime.now() - timedelta(days=7))
with col2:
    st.subheader("Analysis Control")
    st.write("Click the button below to start the analysis.")
    analyze_button = st.button("🚀 Analyze and Generate Report", type="primary")

if analyze_button and uploaded_file and ticker and call_date and models:
    with st.spinner("Analyzing... This may take a few minutes."):
        try:
            transcript = uploaded_file.read().decode("utf-8")
            st.info("✅ Step 1: Cleansing transcript...")
            cleaned_text = clean_text_api(transcript, API_KEY)
            st.info("✅ Step 2: Performing sentiment analysis...")
            sentiment_results = analyze_sentiment(cleaned_text, models)
            st.info("✅ Step 3: Fetching market data...")
            market_change, market_history = get_market_performance(ticker, call_date)
            st.info("✅ Step 4: Generating visualizations...")
            figs = create_visualizations(sentiment_results, market_history, ticker, cleaned_text)
            st.info("✅ Step 5: Generating AI-powered report...")
            report_text = generate_report_api(cleaned_text, sentiment_results, market_change, API_KEY)
            st.info("✅ Step 6: Compiling PDF document...")
            pdf_buffer = create_pdf_report(report_text, figs)
            st.success("Analysis Complete!")
            st.subheader("📊 Analysis Results")
            if 'word_cloud' in figs: st.pyplot(figs['word_cloud'])
            if 'score_comparison' in figs: st.pyplot(figs['score_comparison'])
            if 'market_performance' in figs: st.pyplot(figs['market_performance'])
            st.subheader("📝 Generated Report Summary")
            st.markdown(report_text, unsafe_allow_html=True)
            st.download_button(label="📥 Download Full PDF Report", data=pdf_buffer, file_name=f"{ticker}_ECSA_Report_{call_date}.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            st.code(traceback.format_exc())
elif analyze_button:
    st.error("Please provide all inputs: a transcript file, a ticker, and a date.")
